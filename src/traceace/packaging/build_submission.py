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

import pandas as pd

from ..config import get_config
from ..io import LABEL_COL, load_train
from ..logging_utils import get_logger
from ..paths import models_dir, submission_dir
from ..progress import heartbeat
from ..tasks import task
from .main_template import render_main

log = get_logger("submission.build")


def _collect_boosters(experiment: str) -> list[Any]:
    import lightgbm as lgb

    mdir = models_dir() / experiment.replace(".", "_")
    files = sorted(mdir.glob("fold*.txt"))
    if not files:
        raise FileNotFoundError(
            f"no fold models under {mdir} — run tasks.run('{experiment}') first"
        )
    return [lgb.Booster(model_file=str(f)) for f in files]


def _feature_cols(experiment: str) -> list[str]:
    imp = models_dir() / experiment.replace(".", "_") / "importance.parquet"
    if not imp.is_file():
        raise FileNotFoundError(f"{imp} missing — run the model task first")
    # importance.parquet lists every training feature, in the training order
    return pd.read_parquet(imp)["feature"].tolist()


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
    feature_cols = _feature_cols(experiment)
    calibrator = None
    cal_path = models_dir() / experiment.replace(".", "_") / "calibrator.joblib"
    if cal_path.is_file():
        calibrator = joblib.load(cal_path)

    train_df = load_train()
    bundle = {
        "boosters": boosters,
        "feature_cols": feature_cols,
        "calibrator": calibrator,
        "lo_vectorizer": fit_lo_vectorizer(),
        "lo_prior": _lo_prior(),
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
        "calibrator": (calibrator or {}).get("method", "none"),
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
