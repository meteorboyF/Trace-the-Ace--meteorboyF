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
) -> dict[str, Any]:
    """Compare no-calibration / Platt / isotonic on OOF, using an inner fold loop."""
    import joblib

    oof = load_oof(experiment, subsample=subsample)
    folds = load_folds(subsample=subsample)[["response_id", "fold"]]
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

    # refit the winner on ALL oof data and persist for inference
    if best != "none":
        best_fitter, _ = methods[best]
        assert best_fitter is not None  # "none" is the only entry with a null fitter
        mdir = experiment_dir(experiment, subsample)
        mdir.mkdir(parents=True, exist_ok=True)
        final = best_fitter(p, y)
        # ALSO export the calibrator as plain numbers. The submission ships these, not the
        # fitted object: the runtime's scikit-learn differs from ours, and unpickling across
        # versions warns of "invalid results". Arithmetic is version-proof.
        plain = export_calibrator(best, final)
        joblib.dump(plain, mdir / "calibrator_plain.joblib")
        joblib.dump({"method": best, "model": final}, mdir / "calibrator.joblib")

    cal_exp = f"{experiment}.calibrated"
    out = df[["response_id", "session_id", LABEL_COL]].copy()
    out["pred"] = calibrated[best]
    save_oof(cal_exp, out, subsample=subsample)

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
            for k, v in score_frame(out, cal_exp).items()
            if k in ("logloss", "auc", "delta_vs_lo_only")
        }
    )
    d = runs_dir() / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{experiment.replace('.', '_')}.json").write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(d / f"{experiment.replace('.', '_')}.json")
    return res
