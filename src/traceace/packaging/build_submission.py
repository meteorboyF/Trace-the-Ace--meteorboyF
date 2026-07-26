"""Build ``submission.zip``: main.py at the ROOT of the archive, plus assets.

Archive layout (no wrapping folder — the rules are explicit about this):

    submission.zip
    ├── main.py
    ├── inference_lib.py
    └── assets/
        ├── model.joblib          # boosters + feature_cols + calibrator + lo_vectorizer
        └── MANIFEST.json

The bundle carries everything inference needs and nothing it does not: the per-fold
LightGBM boosters (averaged at predict time), the exact feature column order, the fitted
calibrator, the LO TF-IDF vectorizer (fit on training LO text only), and the per-LO prior
used as a fallback when a transcript cannot be read.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from ..config import get_config
from ..evaluate import experiment_dir
from ..io import LABEL_COL, load_train
from ..logging_utils import get_logger
from ..paths import submission_dir
from ..progress import heartbeat
from ..tasks import task
from .main_template import render_main

log = get_logger("submission.build")


def _collect_boosters(experiment: str) -> list[Any]:
    import lightgbm as lgb

    cfg = get_config()
    # Deliberately the FULL-data directory (subsample=None). A submission must never be
    # built from a subsampled smoke model — see evaluate.experiment_dir.
    mdir = experiment_dir(experiment, None)
    expected = [mdir / f"fold{k}.txt" for k in range(int(cfg.cv["n_splits"]))]
    actual = sorted(mdir.glob("fold*.txt"))
    if actual != expected:
        raise FileNotFoundError(
            f"expected exactly {[p.name for p in expected]} under {mdir}, got "
            f"{[p.name for p in actual]}; retrain {experiment!r} as one complete generation"
        )
    return [lgb.Booster(model_file=str(f)) for f in expected]


def _collect_fold_lo_priors(experiment: str, boosters: list[Any]) -> list[dict[str, Any]]:
    """Load the target-encoding map fitted for each booster's outer training fold."""
    mdir = experiment_dir(experiment, None)
    if not boosters or "lo_prior_enc" not in set(boosters[0].feature_name()):
        return []
    paths = [mdir / f"lo_prior_fold{k}.json" for k in range(len(boosters))]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"fold-specific LO-prior artifacts missing {missing}; retrain {experiment!r} "
            "before building a submission"
        )
    specs = [json.loads(path.read_text()) for path in paths]
    for k, spec in enumerate(specs):
        if int(spec.get("fold", -1)) != k:
            raise RuntimeError(f"{paths[k].name} declares the wrong fold")
    return specs


def _feature_cols(experiment: str, boosters: list[Any]) -> list[str]:
    """The EXACT feature order the boosters were trained on.

    Taken from ``booster.feature_name()`` — the model's own record — because it cannot
    drift from the model by construction.

    **This function previously read ``importance.parquet``, which is sorted by gain
    descending.** That silently shipped a permutation: 179 of 181 positions differed from
    the training order, `main.py` reordered the columns to match, and LightGBM read them
    positionally. Every feature was scrambled. Predictions stayed confident and
    well-formatted, so all 20 verify checks passed — and the leaderboard came back at AUC
    0.4933, below random. Never derive feature order from anything but the model.
    """
    if not boosters:
        raise ValueError("no boosters supplied")
    names = list(boosters[0].feature_name())
    for i, b in enumerate(boosters[1:], start=1):
        if list(b.feature_name()) != names:
            raise RuntimeError(
                f"booster {i} has a different feature order than booster 0 — "
                "the folds were not trained on the same design matrix"
            )
    return names


def _lo_prior(smoothing: float = 20.0) -> dict[str, float]:
    """Per-learning-objective smoothed correctness, for the unreadable-transcript path."""
    df = load_train()
    if "learning_objective_id" not in df.columns:
        return {}
    g = float(df[LABEL_COL].mean())
    agg = df.groupby("learning_objective_id")[LABEL_COL].agg(["sum", "count"])
    sm = (agg["sum"] + smoothing * g) / (agg["count"] + smoothing)
    return {str(k): float(v) for k, v in sm.items()}


@task(
    "submission.build",
    requires="cpu",
    max_tier="cpu",
    description="package main.py + assets into submission.zip (main.py at zip root)",
)
def build(
    experiment: str = "model.gbdt",
    force: bool = False,
    subsample: int | None = None,
    output_name: str = "submission.zip",
) -> dict[str, Any]:
    import joblib

    from ..features.lo_alignment import fit_lo_vectorizer

    cfg = get_config()
    sdir = submission_dir()
    staging = sdir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "assets").mkdir(parents=True, exist_ok=True)

    # --- main.py at ROOT ----------------------------------------------------
    main_py = render_main(seed=cfg.seed, clip_eps=cfg.predict_clip_eps)
    (staging / "main.py").write_text(main_py, encoding="utf-8")

    # --- the shared feature library, copied verbatim (no train/serve skew) ---
    lib_src = Path(__file__).resolve().parent / "inference_lib.py"
    shutil.copyfile(lib_src, staging / "inference_lib.py")

    # --- model bundle -------------------------------------------------------
    boosters = _collect_boosters(experiment)
    feature_cols = _feature_cols(experiment, boosters)
    fold_lo_priors = _collect_fold_lo_priors(experiment, boosters)
    # Ship the PLAIN-NUMBER calibrator, never the fitted sklearn estimator: the runtime's
    # scikit-learn version differs from ours and unpickling across versions warns of
    # "invalid results" (observed in a container smoke test, 2026-07-27).
    calibrator = None
    plain_path = experiment_dir(experiment, None) / "calibrator_plain.joblib"
    if plain_path.is_file():
        calibrator = joblib.load(plain_path)

    train_df = load_train()
    bundle = {
        "boosters": boosters,
        "feature_cols": feature_cols,
        "calibrator": calibrator,
        "lo_vectorizer": fit_lo_vectorizer(),
        "lo_prior": _lo_prior(),
        "lo_prior_by_booster": fold_lo_priors,
        "fallback_prob": float(train_df[LABEL_COL].mean()),
        "seed": cfg.seed,
        "experiment": experiment,
    }
    with heartbeat("writing model bundle"):
        joblib.dump(bundle, staging / "assets" / "model.joblib", compress=3)

    manifest = {
        "experiment": experiment,
        "n_boosters": len(boosters),
        "n_features": len(feature_cols),
        "target_encoding": "fold_specific" if fold_lo_priors else "absent",
        "calibrator": (calibrator or {}).get("method", "none"),
        "sklearn_build_version": __import__("sklearn").__version__,
        "seed": cfg.seed,
        "clip_eps": cfg.predict_clip_eps,
    }
    (staging / "assets" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    # --- zip with main.py at the ROOT ---------------------------------------
    out_zip = sdir / output_name
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(staging))  # relative => no wrapping folder

    size_mb = out_zip.stat().st_size / 1e6
    log.info(
        "submission.build: %s (%.1f MB, %d boosters, %d features)",
        out_zip,
        size_mb,
        len(boosters),
        len(feature_cols),
    )
    return {
        "output_path": str(out_zip),
        "size_mb": round(size_mb, 2),
        "n_boosters": len(boosters),
        "n_features": len(feature_cols),
        "calibrator": manifest["calibrator"],
        "staging_dir": str(staging),
    }
