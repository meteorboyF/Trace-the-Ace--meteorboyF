"""Metrics, OOF persistence, and the anti-goal guard.

Log loss is *the* objective (§10.3) and is headlined everywhere. ROC AUC is reported
for reference only — it does not affect leaderboard rank.

The organizers' explicit anti-goal is predicting correctness from the learning
objective's inferred difficulty without using the transcript. We make that impossible
to ignore structurally: :func:`score_frame` looks up the persisted ``baseline.lo_only``
OOF score and every report carries ``delta_vs_lo_only``. A negative delta means the
model is *worse* than knowing nothing but the learning objective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import get_config
from .io import LABEL_COL, write_parquet
from .logging_utils import get_logger
from .paths import models_dir, oof_dir, runs_dir
from .tasks import task

log = get_logger("evaluate")

BASELINE_KEY = "baseline.lo_only"


def clip_probs(p: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Clip probabilities away from exact 0/1 — log loss is infinite there."""
    if eps is None:
        eps = get_config().predict_clip_eps
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_probs(p)
    return float(log_loss(np.asarray(y, dtype=float), p, labels=[0.0, 1.0]))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p, dtype=float)))


# --- OOF persistence ---------------------------------------------------------
def experiment_name(base: str, subsample: int | None, cv_seed: int | None = None) -> str:
    """Namespace an experiment by subsample size.

    **This exists because of a real bug.** OOF frames were originally keyed by experiment
    name alone, so a 400-session ``selftest`` run silently overwrote the full-data
    ``baseline.lo_only`` OOF. Every subsequent ``delta_vs_lo_only`` was then measured
    against a subsampled baseline — the headline number flipped sign (-0.0078 -> +0.0052)
    with no error raised anywhere. Subsampled results must never share a key with
    full-data results.
    """
    name = base if subsample is None else f"{base}__sub{subsample}"
    return name if cv_seed is None else f"{name}__cv{cv_seed}"


def oof_path(experiment: str) -> Path:
    return oof_dir() / f"{experiment.replace('.', '_')}.parquet"


def experiment_dir(
    experiment: str, subsample: int | None = None, cv_seed: int | None = None
) -> Path:
    """Directory for an experiment's model artifacts, namespaced by subsample.

    Same reasoning as :func:`experiment_name`, but the stakes are higher: ``submission.build``
    reads the fold models from here, so an unnamespaced path let a 400-session self-test
    overwrite the full-data models and would have shipped a submission trained on 1.7% of the
    data — silently, since the files are valid LightGBM boosters either way.
    """
    return models_dir() / experiment_name(experiment, subsample, cv_seed).replace(".", "_")


def save_oof(
    experiment: str,
    df: pd.DataFrame,
    subsample: int | None = None,
    cv_seed: int | None = None,
) -> Path:
    """Persist OOF predictions. Required for calibration and blending (§10.2).

    Pass ``subsample`` so a smoke run cannot overwrite a full-data experiment.
    """
    required = {"response_id", LABEL_COL, "pred"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"OOF frame for {experiment!r} missing columns {missing}")
    name = experiment_name(experiment, subsample, cv_seed)
    path = oof_path(name)
    write_parquet(df[sorted(df.columns)], path)
    log.info("saved OOF for %s -> %s (%d rows)", name, path, len(df))
    return path


def load_oof(
    experiment: str, subsample: int | None = None, cv_seed: int | None = None
) -> pd.DataFrame:
    name = experiment_name(experiment, subsample, cv_seed)
    path = oof_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no OOF for {name!r} at {path} — run that task first")
    return pd.read_parquet(path)


def list_oof() -> list[str]:
    return sorted(p.stem.replace("_", ".", 1) for p in oof_dir().glob("*.parquet"))


# --- scoring -----------------------------------------------------------------
def baseline_logloss(subsample: int | None = None, cv_seed: int | None = None) -> float | None:
    """OOF log loss of ``baseline.lo_only`` at the SAME subsample size.

    Comparing a subsampled model against a full-data baseline (or vice versa) is
    meaningless — different row sets, different achievable loss. Always compare
    like-for-like.
    """
    try:
        b = load_oof(BASELINE_KEY, subsample=subsample, cv_seed=cv_seed)
    except FileNotFoundError:
        return None
    return logloss(b[LABEL_COL].to_numpy(), b["pred"].to_numpy())


def score_frame(
    df: pd.DataFrame,
    experiment: str = "",
    subsample: int | None = None,
    cv_seed: int | None = None,
) -> dict[str, Any]:
    """Score an OOF frame: log loss (headline), AUC, and delta vs baseline.lo_only."""
    y = df[LABEL_COL].to_numpy()
    p = df["pred"].to_numpy()
    ll = logloss(y, p)
    out: dict[str, Any] = {
        "experiment": experiment,
        "logloss": ll,
        "auc": auc(y, p),
        "n": int(len(df)),
        "pos_rate": float(np.mean(y)),
        "subsample": subsample,
        "cv_seed": cv_seed,
    }
    base = baseline_logloss(subsample=subsample, cv_seed=cv_seed)
    if base is not None:
        # negative delta = better than the LO-only bar (lower log loss is better)
        out["baseline_lo_only_logloss"] = base
        out["delta_vs_lo_only"] = ll - base
        out["beats_lo_only"] = bool(ll < base)
        # Guard against the row-count mismatch that made this bug invisible.
        try:
            nb = len(load_oof(BASELINE_KEY, subsample=subsample, cv_seed=cv_seed))
            if nb != len(df):
                out["WARNING_baseline_row_mismatch"] = f"model n={len(df)} vs baseline n={nb}"
                log.error(
                    "delta_vs_lo_only compares %d model rows against %d baseline rows — "
                    "these are NOT comparable. Re-run baseline.lo_only at this subsample.",
                    len(df),
                    nb,
                )
        except FileNotFoundError:
            pass
    return out


# --- slices ------------------------------------------------------------------
def slice_report(
    df: pd.DataFrame,
    slices: dict[str, pd.Series],
    min_n: int = 50,
) -> dict[str, Any]:
    """Per-slice log loss/AUC. ``slices`` maps a name to a per-row category Series."""
    out: dict[str, Any] = {}
    for name, series in slices.items():
        rows = []
        s = series.reindex(df.index)
        for value, idx in df.groupby(s.values).groups.items():
            sub = df.loc[idx]
            if len(sub) < min_n:
                continue
            rows.append(
                {
                    "slice_value": str(value),
                    "n": int(len(sub)),
                    "logloss": logloss(sub[LABEL_COL].to_numpy(), sub["pred"].to_numpy()),
                    "auc": auc(sub[LABEL_COL].to_numpy(), sub["pred"].to_numpy()),
                    "pos_rate": float(sub[LABEL_COL].mean()),
                }
            )
        out[name] = sorted(rows, key=lambda r: -r["n"])
    return out


def quantile_bins(values: pd.Series, n_bins: int = 5, label: str = "q") -> pd.Series:
    """Bin a numeric series into quantiles, returned as readable string labels."""
    try:
        binned = pd.qcut(values, q=n_bins, duplicates="drop")
        return binned.astype(str)
    except (ValueError, TypeError):
        return pd.Series([label] * len(values), index=values.index)


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_pred, frac_positive, count) per probability bin."""
    p = clip_probs(p)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mp, fp, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        mp.append(float(p[m].mean()))
        fp.append(float(y[m].mean()))
        cnt.append(int(m.sum()))
    return np.array(mp), np.array(fp), np.array(cnt)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    mp, fp, cnt = reliability_curve(y, p, n_bins)
    if len(cnt) == 0:
        return float("nan")
    w = cnt / cnt.sum()
    return float(np.sum(w * np.abs(mp - fp)))


@task(
    "evaluate.report",
    requires="cpu",
    max_tier="cpu",
    description="log loss, AUC, calibration and per-slice performance for an experiment",
)
def report(
    experiment: str = "model.gbdt",
    force: bool = False,
    subsample: int | None = None,
) -> dict[str, Any]:
    """Score a persisted OOF experiment and write a JSON report."""
    df = load_oof(experiment, subsample=subsample)
    res = score_frame(df, experiment, subsample=subsample)
    res["ece"] = expected_calibration_error(df[LABEL_COL].to_numpy(), df["pred"].to_numpy())

    # slices, where the necessary columns are available on the OOF frame
    slices: dict[str, pd.Series] = {}
    for col, name in [
        ("n_tokens", "transcript_length"),
        ("n_utterances", "turn_count"),
        ("student_talk_ratio", "student_talk_ratio"),
    ]:
        if col in df.columns:
            slices[name] = quantile_bins(df[col])
    if "learning_objective_id" in df.columns:
        slices["learning_objective"] = df["learning_objective_id"].astype(str)
    if slices:
        res["slices"] = slice_report(df, slices)

    d = runs_dir() / "evaluate"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{experiment.replace('.', '_')}.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(path)

    delta = res.get("delta_vs_lo_only")
    log.info(
        "evaluate %s: logloss=%.5f auc=%.4f%s",
        experiment,
        res["logloss"],
        res["auc"],
        f" delta_vs_lo_only={delta:+.5f}" if delta is not None else "",
    )
    return res
