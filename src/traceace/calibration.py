"""Probability calibration — isotonic and Platt, fit on OOF predictions.

The organizers explicitly note that log loss "can often be improved with calibration",
and log loss is far more responsive to calibration than to ranking. A model with
mediocre AUC but honest probabilities can beat a sharper model that is overconfident.

Both calibrators are fit **on out-of-fold predictions only** and evaluated with an inner
cross-fold loop, so the reported improvement is not itself overfit. The winner (by OOF
log loss, including "none") is persisted and applied at inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cv import load_folds
from .evaluate import (
    clip_probs,
    expected_calibration_error,
    experiment_dir,
    load_oof,
    logloss,
    save_oof,
    score_frame,
)
from .io import LABEL_COL
from .logging_utils import get_logger
from .paths import runs_dir
from .progress import pbar
from .tasks import task

log = get_logger("calibration")


def fit_platt(p: np.ndarray, y: np.ndarray):
    """Logistic regression on the logit of p (Platt scaling)."""
    from sklearn.linear_model import LogisticRegression

    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return lr


def apply_platt(model, p: np.ndarray) -> np.ndarray:
    z = _logit(p).reshape(-1, 1)
    return model.predict_proba(z)[:, 1]


def fit_isotonic(p: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def apply_isotonic(model, p: np.ndarray) -> np.ndarray:
    return model.predict(p)


def export_calibrator(method: str, model: Any) -> dict[str, Any]:
    """Reduce a fitted calibrator to plain numbers for version-proof serialization.

    See ``inference_lib.apply_calibration`` for the matching arithmetic. Keeping sklearn
    objects out of the shipped bundle removes a whole class of cross-version risk.
    """
    if method == "platt":
        return {
            "method": "platt",
            "coef": float(np.ravel(model.coef_)[0]),
            "intercept": float(np.ravel(model.intercept_)[0]),
        }
    if method == "isotonic":
        return {
            "method": "isotonic",
            "x_thresholds": np.asarray(model.X_thresholds_, dtype=float).tolist(),
            "y_thresholds": np.asarray(model.y_thresholds_, dtype=float).tolist(),
        }
    return {"method": "none"}


def _persist_calibrator(method: str, model: Any, model_dir: Path) -> None:
    """Persist exactly the selected calibrator, removing artifacts from older runs."""
    import joblib

    model_dir.mkdir(parents=True, exist_ok=True)
    plain_path = model_dir / "calibrator_plain.joblib"
    fitted_path = model_dir / "calibrator.joblib"
    if method == "none":
        # A previous run may have selected Platt/isotonic. Leaving either file behind
        # makes submission.build silently ship that stale calibrator even though this run
        # selected "none".
        plain_path.unlink(missing_ok=True)
        fitted_path.unlink(missing_ok=True)
        return

    plain = export_calibrator(method, model)
    joblib.dump(plain, plain_path)
    joblib.dump({"method": method, "model": model}, fitted_path)


def _logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


@task(
    "calibrate.fit",
    requires="cpu",
    max_tier="cpu",
    description="fit isotonic + Platt on OOF; keep whichever wins on log loss",
)
def fit(
    experiment: str = "model.gbdt",
    force: bool = False,
    subsample: int | None = None,
    cv_seed: int | None = None,
) -> dict[str, Any]:
    """Compare no-calibration / Platt / isotonic on OOF, using an inner fold loop."""
    oof = load_oof(experiment, subsample=subsample, cv_seed=cv_seed)
    folds = load_folds(subsample=subsample, cv_seed=cv_seed)[["response_id", "fold"]]
    df = oof.merge(folds, on="response_id", how="left")
    if df["fold"].isna().any():
        raise RuntimeError("OOF rows missing fold assignment — rebuild cv or the OOF frame")

    y = df[LABEL_COL].to_numpy(dtype=float)
    p = df["pred"].to_numpy(dtype=float)

    methods = {
        "none": (None, None),
        "platt": (fit_platt, apply_platt),
        "isotonic": (fit_isotonic, apply_isotonic),
    }
    results: dict[str, dict[str, float]] = {}
    calibrated: dict[str, np.ndarray] = {}

    for name, (fitter, applier) in methods.items():
        if fitter is None or applier is None:
            # the "none" variant: pass the raw predictions through unchanged
            cal = p.copy()
        else:
            cal = np.full(len(df), np.nan)
            # inner cross-fold: fit the calibrator out-of-fold so the gain is honest
            for k in pbar(
                sorted(df["fold"].unique()), desc=f"calibrate:{name}", unit="fold", leave=False
            ):
                tr = (df["fold"] != k).to_numpy()
                va = (df["fold"] == k).to_numpy()
                model = fitter(p[tr], y[tr])
                cal[va] = applier(model, p[va])
        cal = clip_probs(cal)
        calibrated[name] = cal
        results[name] = {
            "logloss": logloss(y, cal),
            "ece": expected_calibration_error(y, cal),
        }

    best = min(results, key=lambda k: results[k]["logloss"])
    log.info(
        "calibration for %s: %s -> best=%s",
        experiment,
        {k: round(v["logloss"], 5) for k, v in results.items()},
        best,
    )

    # Refit the winner on ALL OOF data and persist exactly that selection. This also
    # removes a stale calibrator when "none" wins after an earlier calibrated run.
    mdir = experiment_dir(experiment, subsample, cv_seed)
    if best != "none":
        best_fitter, _ = methods[best]
        assert best_fitter is not None  # "none" is the only entry with a null fitter
        final = best_fitter(p, y)
        _persist_calibrator(best, final, mdir)
    else:
        _persist_calibrator("none", None, mdir)

    cal_exp = f"{experiment}.calibrated"
    out = df[["response_id", "session_id", LABEL_COL]].copy()
    out["pred"] = calibrated[best]
    save_oof(cal_exp, out, subsample=subsample, cv_seed=cv_seed)

    res: dict[str, Any] = {
        "experiment": experiment,
        "calibrated_experiment": cal_exp,
        "methods": results,
        "best_method": best,
        "improvement": results["none"]["logloss"] - results[best]["logloss"],
    }
    res.update(
        {
            f"final_{k}": v
            for k, v in score_frame(out, cal_exp, subsample=subsample, cv_seed=cv_seed).items()
            if k in ("logloss", "auc", "delta_vs_lo_only")
        }
    )
    d = runs_dir() / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{experiment.replace('.', '_')}.json").write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(d / f"{experiment.replace('.', '_')}.json")
    return res


# ---------------------------------------------------------------------------
# Deployment shrinkage — calibrating for the regime we actually deploy into.
# ---------------------------------------------------------------------------
# Our CV regime is EASIER than the test regime: CV validation rows almost always have a
# learning objective the model has seen (99.7% of them), and every session comes from one
# provider. The organizers confirmed the test set has objectives absent from training and
# is "not drawn entirely from Third Space Learning". A model tuned on the easy regime is
# therefore systematically OVERCONFIDENT when deployed.
#
# The leaderboard showed exactly that signature: AUROC 0.6014 (real ranking signal) with
# log loss 0.6106 (marginally WORSE than predicting the global base rate). Ranking works;
# the magnitudes are wrong.
#
# Shrinking predictions toward the base rate, p' = base + w*(p - base), cannot change
# ranking at all (AUC is invariant) but recovers a large amount of log loss when the model
# is overconfident. On the unseen-objective holdout — our closest in-house proxy for test
# conditions — the optimum is w≈0.5, worth ~0.007 log loss.
#
# w is fitted on TRAINING data (the unseen-objective holdout). It is NOT tuned against
# leaderboard feedback, which would be rules-adjacent.


def fit_shrinkage(y: np.ndarray, p: np.ndarray, base_rate: float) -> float:
    """Find the shrinkage weight minimizing log loss on a holdout matched to deployment."""
    grid = np.arange(0.05, 1.01, 0.05)
    losses = [logloss(y, clip_probs(base_rate + w * (p - base_rate))) for w in grid]
    return float(grid[int(np.argmin(losses))])


def apply_shrinkage(p: np.ndarray, weight: float, base_rate: float) -> np.ndarray:
    """p' = base + w*(p - base). Ranking-preserving; only the magnitudes move."""
    return clip_probs(base_rate + float(weight) * (np.asarray(p, dtype=float) - base_rate))
