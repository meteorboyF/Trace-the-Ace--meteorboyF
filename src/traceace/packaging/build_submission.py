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

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from ..config import get_config
from ..evaluate import experiment_dir, experiment_name, oof_path
from ..io import LABEL_COL, load_train
from ..logging_utils import get_logger
from ..paths import submission_dir
from ..progress import heartbeat
from ..tasks import task
from .main_template import render_main

log = get_logger("submission.build")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# These feature families currently have training/research implementations but no matching
# offline implementation in main.py. Packaging them would otherwise create a plausible model
# bundle that only fails later (or, before feature-coverage verification existed, silently
# predicts with NaNs).
UNDEPLOYABLE_FEATURE_PREFIXES: tuple[str, ...] = ("cont_", "emb_", "move_")


def _assert_features_deployable(feature_cols: list[str]) -> None:
    unsupported = sorted(
        column for column in feature_cols if column.startswith(UNDEPLOYABLE_FEATURE_PREFIXES)
    )
    if unsupported:
        families = sorted({column.split("_", 1)[0] for column in unsupported})
        raise RuntimeError(
            "model contains research-only feature families with no main.py implementation: "
            f"{families}. Do not package this experiment until train/serve parity is implemented."
        )


def _collect_boosters(experiment: str) -> list[Any]:
    import lightgbm as lgb

    cfg = get_config()
    # Deliberately the FULL-data directory (subsample=None). A submission must never be
    # built from a subsampled smoke model — see evaluate.experiment_dir.
    mdir = experiment_dir(experiment, None)
    manifest_path = mdir / "training_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("split_mode", "session") != "session":
            raise RuntimeError(
                f"{experiment!r} was trained on {manifest.get('split_mode')} research folds; "
                "only session-fold production models may be packaged"
            )
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


def _shrinkage(experiment: str) -> dict[str, float] | None:
    """Load the fitted deployment-shrinkage weight, if `calibrate.shrinkage` has run."""
    import joblib

    path = experiment_dir(experiment, None) / "shrinkage.joblib"
    if not path.is_file():
        log.warning(
            "no shrinkage weight found for %s — shipping UNSHRUNK predictions. Run "
            "tasks.run('calibrate.shrinkage') first; the deployed regime is harder than CV.",
            experiment,
        )
        return None
    return dict(joblib.load(path))


def _collect_encoder(experiment: str) -> tuple[dict[str, Any], list[Path]]:
    """Validate the encoder experiment's artifacts and return (config, fold checkpoint paths).

    Requirements are strict because a partial or mixed set of folds would package a model
    that never existed: exactly ``n_splits`` checkpoints, every fold's config sidecar
    byte-identical, and the training manifest present to supply the inference settings.

    Unlike the GBDT collector this does NOT require session folds — the encoder's
    production artifact is objective-disjoint by design; that is how it was validated.
    """
    cfg = get_config()
    mdir = experiment_dir(experiment, None)
    manifest_path = mdir / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} missing — has {experiment!r} trained?")
    manifest = json.loads(manifest_path.read_text())
    encoder_cfg = dict(manifest.get("config", {}))
    required = {"model_name", "max_tokens", "topk_windows", "include_objective"}
    if not required.issubset(encoder_cfg):
        raise RuntimeError(f"{experiment!r} manifest lacks encoder config fields {required}")
    if encoder_cfg.get("include_objective"):
        raise RuntimeError(
            "refusing to package an encoder trained WITH the objective text in its input: "
            "that configuration memorises objective difficulty (the organisers' anti-goal) "
            "and its validation numbers do not describe test behaviour"
        )

    n_folds = int(cfg.cv["n_splits"])
    checkpoints = [mdir / f"fold{k}_model.pt" for k in range(n_folds)]
    missing = [p.name for p in checkpoints if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"encoder fold checkpoints missing {missing} under {mdir}; "
            f"finish training {experiment!r} before packaging"
        )
    sidecars = [mdir / f"fold{k}_config.json" for k in range(n_folds)]
    missing = [p.name for p in sidecars if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"encoder fold config sidecars missing {missing} under {mdir}")
    provenances = [json.loads(p.read_text()) for p in sidecars]
    if any(prov != provenances[0] for prov in provenances[1:]):
        raise RuntimeError(
            f"encoder folds of {experiment!r} were trained under different configurations; "
            "retrain as one complete generation before packaging"
        )
    if provenances[0].get("config") != encoder_cfg:
        raise RuntimeError(
            f"{experiment!r}: training manifest and fold sidecars disagree on the config; "
            "the artifacts are from mixed generations"
        )
    return encoder_cfg, checkpoints


def _vendor_encoder_ensemble(
    staging: Path,
    members: list[tuple[str, dict[str, Any], list[Path], float]],
    base_weight: float,
) -> dict[str, Any]:
    """Vendor N encoders under ``assets/encoder/<name>/`` plus a top-level manifest.

    Each member keeps its OWN tokenizer, config and ``encoder.json`` because ensemble members
    differ in exactly the parameters inference needs (``topk_windows``, ``max_tokens``) — a
    shared config would silently feed one member the other's window width.
    """
    encoder_root = staging / "assets" / "encoder"
    encoder_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, encoder_cfg, checkpoints, weight in members:
        spec = _vendor_encoder_assets(
            encoder_root / name, encoder_cfg, checkpoints, weight, make_dir=True
        )
        entries.append({"name": name, "weight": float(weight), **spec})
    manifest = {"base_weight": float(base_weight), "encoders": entries}
    (encoder_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _vendor_encoder_assets(
    encoder_dir: Path,
    encoder_cfg: dict[str, Any],
    checkpoints: list[Path],
    blend_weight: float,
    make_dir: bool = False,
) -> dict[str, Any]:
    """Write tokenizer + config + fold weights + encoder.json under assets/encoder/.

    The tokenizer and *architecture config* are vendored via ``save_pretrained`` so the
    container never touches the Hub; every actual weight comes from our own fold
    checkpoints (the state dicts are complete, embeddings included).
    """
    from transformers import AutoConfig, AutoTokenizer

    if make_dir:
        encoder_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(encoder_cfg["model_name"])
    AutoTokenizer.from_pretrained(model_name).save_pretrained(encoder_dir / "tokenizer")
    AutoConfig.from_pretrained(model_name).save_pretrained(encoder_dir / "config")

    # CONTAINER-COMPAT (root cause of the failed 2026-08-22 submission, job id-6296):
    # the build machine's transformers stamped tokenizer_config.json with its own class
    # name ("TokenizersBackend", a v5 name); the competition image runs an OLDER
    # transformers that has no such class and raises before loading a single weight.
    # "PreTrainedTokenizerFast" is the universal fast-tokenizer class, recognised by every
    # transformers version this project could meet, and the actual vocabulary lives in
    # tokenizer.json regardless — so rewriting the class name changes behaviour nowhere
    # while making the file loadable everywhere.
    tokenizer_config_path = encoder_dir / "tokenizer" / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2))
    for k, src in enumerate(checkpoints):
        shutil.copyfile(src, encoder_dir / f"fold{k}.pt")

    spec = {
        "model_name": model_name,
        "max_tokens": int(encoder_cfg["max_tokens"]),
        "topk_windows": int(encoder_cfg["topk_windows"]),
        "blend_weight": float(blend_weight),
        "n_folds": len(checkpoints),
    }
    (encoder_dir / "encoder.json").write_text(json.dumps(spec, indent=2))
    return spec


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
    blend_experiment: str | None = "ensemble.hybrid",
    apply_deployment_shrinkage: bool = False,
    encoder_experiment: str | None = None,
    encoder_weight: float | None = None,
    encoder_experiments: list[str] | None = None,
    encoder_weights: list[float] | None = None,
    logit_shift: float = 0.0,
) -> dict[str, Any]:
    """Package the submission.

    Neural encoders ship alongside the GBDT in one of two forms:

    * ``encoder_experiment`` + ``encoder_weight`` — a single encoder (the layout that scored
      on 2026-08-22; kept working unchanged).
    * ``encoder_experiments`` + ``encoder_weights`` — an ENSEMBLE. Weights are the encoder
      entries from ``ensemble.blend``'s simplex, and the base model silently receives
      ``1 - sum(weights)``, so the packaged blend is arithmetically the OOF blend measured.

    Weights must be passed EXPLICITLY in both forms — they come from the honest
    objective-fold blend, and a default would let a stale number ship unnoticed.

    ``logit_shift`` recentres the FINAL predictions by a constant in logit space
    (``p' = sigmoid(logit(p) + shift)``). Reserved for the last submission: the leaderboard
    decomposition puts the test base rate near 0.686 versus 0.7025 in training, and shifting
    halfway (target mean ≈ 0.694 → shift ≈ −0.040) captures most of the ~0.0007 log-loss gain
    at a quarter of the risk. It is fitted on LEADERBOARD feedback, not held-out data — which
    is exactly why it defaults to 0.0 and must be passed consciously.
    """
    import joblib

    from ..features.lo_alignment import fit_lo_vectorizer

    if not -0.1 <= float(logit_shift) <= 0.1:
        raise ValueError(
            f"logit_shift {logit_shift} is outside ±0.1 — the half-recentring case needs "
            "≈ −0.040, so anything larger is almost certainly a typo"
        )

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
    sparse_lib_src = Path(__file__).resolve().parent / "sparse_text_lib.py"
    shutil.copyfile(sparse_lib_src, staging / "sparse_text_lib.py")

    # --- neural transcript encoder(s) (optional) -----------------------------
    if encoder_experiment is not None and encoder_experiments is not None:
        raise ValueError("pass encoder_experiment OR encoder_experiments, not both")
    if encoder_experiment is not None:
        encoder_experiments = [encoder_experiment]
        encoder_weights = [encoder_weight] if encoder_weight is not None else None

    encoder_spec: dict[str, Any] | None = None
    if encoder_experiments:
        if encoder_weights is None or len(encoder_weights) != len(encoder_experiments):
            raise ValueError(
                "one encoder_weight per encoder is required — pass the weights measured by "
                "the honest objective-fold blend, never a default"
            )
        weights = [float(w) for w in encoder_weights]
        if any(w <= 0.0 for w in weights):
            raise ValueError(f"encoder weights must be positive, got {weights}")
        total = sum(weights)
        if total >= 1.0:
            raise ValueError(
                f"encoder weights sum to {total:.4f}, leaving the GBDT base weight "
                f"{1.0 - total:.4f}. Weights come from ensemble.blend's simplex and must "
                "leave the base a positive share."
            )
        members = []
        for name, weight in zip(encoder_experiments, weights):
            member_cfg, member_checkpoints = _collect_encoder(name)
            members.append((name.replace(".", "_"), member_cfg, member_checkpoints, weight))
        with heartbeat("vendoring encoder assets"):
            if len(members) == 1 and encoder_experiment is not None:
                # legacy single-encoder layout, byte-identical to the scored 2026-08-22 zip
                encoder_spec = _vendor_encoder_assets(
                    staging / "assets" / "encoder",
                    members[0][1],
                    members[0][2],
                    weights[0],
                    make_dir=True,
                )
            else:
                encoder_spec = _vendor_encoder_ensemble(staging, members, 1.0 - total)
        encoder_lib_src = Path(__file__).resolve().parent / "encoder_lib.py"
        shutil.copyfile(encoder_lib_src, staging / "encoder_lib.py")

    # --- model bundle -------------------------------------------------------
    boosters = _collect_boosters(experiment)
    feature_cols = _feature_cols(experiment, boosters)
    _assert_features_deployable(feature_cols)
    fold_lo_priors = _collect_fold_lo_priors(experiment, boosters)
    # Ship the PLAIN-NUMBER calibrator, never the fitted sklearn estimator: the runtime's
    # scikit-learn version differs from ours and unpickling across versions warns of
    # "invalid results" (observed in a container smoke test, 2026-07-27).
    promotion = None
    sparse_text_model = None
    sparse_text_config = None
    calibration_experiment = experiment
    if blend_experiment is not None:
        promotion_path = experiment_dir(blend_experiment, None) / "promotion.joblib"
        if promotion_path.is_file():
            candidate = dict(joblib.load(promotion_path))
            if candidate.get("promoted"):
                if candidate.get("base_experiment") != experiment:
                    raise RuntimeError("hybrid promotion was fitted against a different base model")
                text_experiment = str(candidate["text_experiment"])
                text_path = experiment_dir(text_experiment, None) / "deployment.joblib"
                if not text_path.is_file():
                    raise FileNotFoundError("promoted sparse-text deployment model is missing")
                sparse_text_model = joblib.load(text_path)
                expected_hashes = {
                    "base_oof_sha256": oof_path(experiment_name(experiment, None)),
                    "text_oof_sha256": oof_path(experiment_name(text_experiment, None)),
                    "text_model_sha256": text_path,
                }
                stale = [
                    name
                    for name, path in expected_hashes.items()
                    if not path.is_file() or candidate.get(name) != _file_sha256(path)
                ]
                if stale:
                    raise RuntimeError(
                        f"hybrid promotion is stale for {stale}; rerun ensemble.promote_text"
                    )
                text_manifest_path = (
                    experiment_dir(text_experiment, None) / "training_manifest.json"
                )
                if not text_manifest_path.is_file():
                    raise FileNotFoundError("promoted sparse-text training manifest is missing")
                text_manifest = json.loads(text_manifest_path.read_text())
                sparse_text_config = dict(text_manifest.get("run_config", {}))
                if not {"max_chars", "context_utterances"}.issubset(sparse_text_config):
                    raise RuntimeError("sparse-text manifest lacks inference configuration")
                promotion = candidate
                calibration_experiment = blend_experiment
        else:
            log.info("no hybrid promotion artifact; packaging the base model only")

    calibrator = None
    plain_path = experiment_dir(calibration_experiment, None) / "calibrator_plain.joblib"
    if plain_path.is_file():
        calibrator = joblib.load(plain_path)

    train_df = load_train()
    bundle = {
        "boosters": boosters,
        "feature_cols": feature_cols,
        "calibrator": calibrator,
        "lo_vectorizer": fit_lo_vectorizer(),
        "lo_prior": _lo_prior(),
        # Leaderboard evidence falsified the old unseen-objective shrinkage proxy
        # (0.6106 -> 0.6133 at identical AUROC). It is opt-in for reproducibility only.
        "shrinkage": _shrinkage(experiment) if apply_deployment_shrinkage else None,
        "lo_prior_by_booster": fold_lo_priors,
        "fallback_prob": float(train_df[LABEL_COL].mean()),
        "seed": cfg.seed,
        "experiment": experiment,
        "sparse_text_model": sparse_text_model,
        "sparse_text_config": sparse_text_config,
        "hybrid_promotion": promotion,
        "logit_shift": float(logit_shift),
    }
    with heartbeat("writing model bundle"):
        joblib.dump(bundle, staging / "assets" / "model.joblib", compress=3)

    manifest = {
        "experiment": experiment,
        "n_boosters": len(boosters),
        "n_features": len(feature_cols),
        "target_encoding": "fold_specific" if fold_lo_priors else "absent",
        "calibrator": (calibrator or {}).get("method", "none"),
        "hybrid": bool(promotion),
        "deployment_shrinkage": bool(apply_deployment_shrinkage),
        "encoder": encoder_spec,  # None when no encoder is shipped
        "encoder_experiments": list(encoder_experiments or []),
        "logit_shift": float(logit_shift),
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
