"""Blending — log-loss-optimal weights over persisted OOF predictions.

Weights are found by constrained optimization (simplex: non-negative, summing to 1) on
the OOF matrix, minimizing log loss directly rather than a proxy. Blending happens in
**logit space**, which is better behaved for probability averaging than the arithmetic
mean and keeps the result calibrated when the inputs are.

Only experiments with matching ``response_id`` sets can be blended, which is guaranteed
because every model writes OOF over the same persisted folds.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .evaluate import clip_probs, load_oof, logloss, save_oof, score_frame
from .io import LABEL_COL
from .logging_utils import get_logger
from .paths import models_dir, runs_dir
from .tasks import task

log = get_logger("ensemble")


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

    from .evaluate import list_oof

    if experiments is None:
        # everything except the baselines and previous blends
        experiments = [
            e for e in list_oof() if not e.startswith("baseline.") and not e.startswith("ensemble.")
        ]
    if len(experiments) == 0:
        raise RuntimeError("no OOF experiments to blend — train a model first")

    frames = {e: load_oof(e).set_index("response_id") for e in experiments}
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if not common:
        raise RuntimeError("no common response_ids across the requested experiments")

    preds = np.column_stack([frames[e].loc[common, "pred"].to_numpy() for e in experiments])
    first = frames[experiments[0]].loc[common]
    y = first[LABEL_COL].to_numpy(dtype=float)

    w = optimal_weights(preds, y)
    blended = blend_logit(preds, w)

    out = pd.DataFrame(
        {
            "response_id": common,
            "session_id": first["session_id"].to_numpy(),
            LABEL_COL: y,
            "pred": blended,
        }
    )
    save_oof(output_experiment, out)

    mdir = models_dir() / output_experiment.replace(".", "_")
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"experiments": experiments, "weights": w.tolist()}, mdir / "weights.joblib")

    res = score_frame(out, output_experiment)
    res.update(
        {
            "experiments": experiments,
            "weights": {e: float(wi) for e, wi in zip(experiments, w)},
            "individual_logloss": {e: logloss(y, preds[:, i]) for i, e in enumerate(experiments)},
            "n_common": len(common),
        }
    )
    d = runs_dir() / "ensemble"
    d.mkdir(parents=True, exist_ok=True)
    (d / "blend.json").write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(d / "blend.json")
    log.info("blend: logloss=%.5f weights=%s", res["logloss"], res["weights"])
    return res
