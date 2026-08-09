"""Blending — log-loss-optimal weights over persisted OOF predictions.

Weights are found by constrained optimization (simplex: non-negative, summing to 1) on
the OOF matrix, minimizing log loss directly rather than a proxy. Blending happens in
**logit space**, which is better behaved for probability averaging than the arithmetic
mean and keeps the result calibrated when the inputs are.

Only experiments with matching ``response_id`` sets can be blended, which is guaranteed
because every model writes OOF over the same persisted folds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .evaluate import (
    align_compatible_oof,
    auc,
    clip_probs,
    experiment_dir,
    experiment_name,
    load_oof,
    logloss,
    oof_path,
    save_oof,
    score_frame,
)
from .io import LABEL_COL
from .logging_utils import get_logger
from .paths import runs_dir
from .tasks import task

log = get_logger("ensemble")


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def blend_logit(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average in logit space -> probability."""
    return _sigmoid(_logit(preds) @ weights)


def optimal_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find simplex weights minimizing log loss. ``preds`` is (n_samples, n_models)."""
    n_models = preds.shape[1]
    if n_models == 1:
        return np.ones(1)

    def obj(w: np.ndarray) -> float:
        w = np.abs(w)
        s = w.sum()
        if s <= 0:
            return 1e9
        return logloss(y, blend_logit(preds, w / s))

    x0 = np.ones(n_models) / n_models
    res = minimize(
        obj, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6}
    )
    w = np.abs(res.x)
    return w / w.sum() if w.sum() > 0 else x0


@task(
    "ensemble.blend",
    requires="cpu",
    max_tier="cpu",
    description="log-loss-optimal blend weights over OOF predictions",
)
def blend(
    experiments: list[str] | None = None,
    force: bool = False,
    subsample: int | None = None,
    output_experiment: str = "ensemble.blend",
) -> dict[str, Any]:
    import joblib

    if experiments is None:
        raise ValueError(
            "experiments must be explicit; auto-discovery mixes full, subsampled, "
            "repeated-seed, calibrated, and ablation OOF artifacts"
        )
    if len(experiments) == 0:
        raise RuntimeError("no OOF experiments to blend — train a model first")

    raw = {e: load_oof(e, subsample=subsample) for e in experiments}
    first = raw[experiments[0]].reset_index(drop=True)
    frames = {
        experiments[0]: first,
        **{e: align_compatible_oof(first, raw[e], experiments[0], e) for e in experiments[1:]},
    }
    preds = np.column_stack([frames[e]["pred"].to_numpy() for e in experiments])
    y = first[LABEL_COL].to_numpy(dtype=float)

    w = optimal_weights(preds, y)
    blended = blend_logit(preds, w)

    out = pd.DataFrame(
        {
            "response_id": first["response_id"].to_numpy(),
            "session_id": first["session_id"].to_numpy(),
            LABEL_COL: y,
            "pred": blended,
        }
    )
    save_oof(output_experiment, out, subsample=subsample)

    mdir = experiment_dir(output_experiment, subsample)
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"experiments": experiments, "weights": w.tolist()}, mdir / "weights.joblib")

    res = score_frame(out, output_experiment, subsample=subsample)
    res.update(
        {
            "experiments": experiments,
            "weights": {e: float(wi) for e, wi in zip(experiments, w)},
            "individual_logloss": {e: logloss(y, preds[:, i]) for i, e in enumerate(experiments)},
            "n_common": len(first),
        }
    )
    d = runs_dir() / "ensemble"
    d.mkdir(parents=True, exist_ok=True)
    (d / "blend.json").write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(d / "blend.json")
    log.info("blend: logloss=%.5f weights=%s", res["logloss"], res["weights"])
    return res


@task(
    "ensemble.promote_text",
    requires="cpu",
    max_tier="cpu",
    description="cross-fitted promotion gate for sparse text + production GBDT",
)
def promote_text(
    force: bool = False,
    subsample: int | None = None,
    base_experiment: str = "model.gbdt",
    text_experiment: str = "model.sparse_text",
    output_experiment: str = "ensemble.hybrid",
    min_gain: float = 0.0003,
) -> dict[str, Any]:
    """Promote text only when fold-held-out blending beats the production model.

    Blend weights are selected on four folds and evaluated on the untouched fifth fold.
    The deployment weight is the median of those five choices, which is deliberately more
    conservative than optimizing one weight on all OOF labels.
    """
    import joblib

    from .cv import load_folds

    base = load_oof(base_experiment, subsample=subsample).reset_index(drop=True)
    text_frame = align_compatible_oof(
        base, load_oof(text_experiment, subsample=subsample), base_experiment, text_experiment
    )
    fold_table = load_folds(subsample=subsample)[["response_id", "fold"]]
    work = base.merge(fold_table, on="response_id", how="left", validate="one_to_one")
    if work["fold"].isna().any():
        raise RuntimeError("hybrid promotion rows are missing fold assignments")
    y = work[LABEL_COL].to_numpy(dtype=float)
    base_p = base["pred"].to_numpy(dtype=float)
    text_p = text_frame["pred"].to_numpy(dtype=float)
    candidates = np.linspace(0.0, 0.5, 11)
    honest = np.full(len(base), np.nan, dtype=float)
    selected: list[float] = []
    for fold in sorted(int(value) for value in work["fold"].unique()):
        valid = work["fold"].eq(fold).to_numpy()
        train = ~valid
        losses = [
            logloss(
                y[train],
                blend_logit(
                    np.column_stack([base_p[train], text_p[train]]), np.array([1 - weight, weight])
                ),
            )
            for weight in candidates
        ]
        weight = float(candidates[int(np.argmin(losses))])
        selected.append(weight)
        honest[valid] = blend_logit(
            np.column_stack([base_p[valid], text_p[valid]]), np.array([1 - weight, weight])
        )
    if not np.isfinite(honest).all():
        raise RuntimeError("hybrid promotion left rows unpredicted")

    base_loss = logloss(y, base_p)
    honest_loss = logloss(y, honest)
    text_auc = auc(y, text_p)
    promoted = bool(honest_loss <= base_loss - min_gain and text_auc > 0.5)
    deployment_weight = float(np.median(selected)) if promoted else 0.0
    output = base[["response_id", "session_id", LABEL_COL]].copy()
    output["pred"] = honest if promoted else base_p
    save_oof(output_experiment, output, subsample=subsample)

    spec = {
        "base_experiment": base_experiment,
        "text_experiment": text_experiment,
        "promoted": promoted,
        "deployment_text_weight": deployment_weight,
        "fold_text_weights": selected,
        "min_gain": min_gain,
        "base_logloss": base_loss,
        "honest_hybrid_logloss": honest_loss,
        "honest_gain": base_loss - honest_loss,
        "text_logloss": logloss(y, text_p),
        "text_auc": text_auc,
        "base_oof_sha256": _file_sha256(oof_path(experiment_name(base_experiment, subsample))),
        "text_oof_sha256": _file_sha256(oof_path(experiment_name(text_experiment, subsample))),
        "text_model_sha256": _file_sha256(
            experiment_dir(text_experiment, subsample) / "deployment.joblib"
        ),
    }
    mdir = experiment_dir(output_experiment, subsample)
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(spec, mdir / "promotion.joblib")
    (mdir / "promotion.json").write_text(json.dumps(spec, indent=2))
    # Any calibrator here belongs to the previous hybrid OOF generation.
    (mdir / "calibrator.joblib").unlink(missing_ok=True)
    (mdir / "calibrator_plain.joblib").unlink(missing_ok=True)
    result = score_frame(output, output_experiment, subsample=subsample)
    result.update(spec)
    result["output_path"] = str(mdir / "promotion.json")
    log.info(
        "hybrid promotion: promoted=%s gain=%.5f weight=%.2f",
        promoted,
        spec["honest_gain"],
        deployment_weight,
    )
    return result
