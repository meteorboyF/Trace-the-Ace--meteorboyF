"""Baselines — the floor and the bar.

* ``baseline.prior``   — predict the global base rate for everything. The floor.
  With a measured positive rate of 0.7025 the expected log loss is the label entropy,
  ≈ **0.609**. Anything at or above this is worthless.

* ``baseline.lo_only`` — predict the smoothed per-learning-objective mean correctness,
  using **no transcript information whatsoever**. This is deliberately the organizers'
  stated *anti-goal* implemented as a baseline, so that every subsequent model is scored
  against it. A transcript model that cannot beat this has not learned anything about
  tutoring — it has only learned which topics are hard.

Both are fit strictly **within each CV fold** (per-LO means come from training folds
only) so the comparison is honest and the OOF frames are blendable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..cv import load_folds
from ..evaluate import save_oof, score_frame
from ..io import LABEL_COL, load_train
from ..logging_utils import get_logger
from ..progress import pbar
from ..tasks import task

log = get_logger("baseline")

LO_COL = "learning_objective_id"


def _train_with_folds(subsample: int | None) -> pd.DataFrame:
    """Join training features with persisted folds."""
    folds = load_folds(subsample=subsample)
    feats = load_train()
    df = folds.merge(
        feats.drop(columns=[c for c in (LABEL_COL,) if c in feats.columns]),
        on=["response_id", "session_id"],
        how="left",
    )
    return df


@task(
    "baseline.prior",
    requires="cpu",
    max_tier="cpu",
    description="global base rate — the log loss floor",
)
def prior(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    df = _train_with_folds(subsample)
    df = df.copy()
    df["pred"] = np.nan

    for k in pbar(sorted(df["fold"].unique()), desc="baseline.prior folds", unit="fold"):
        tr = df["fold"] != k
        va = df["fold"] == k
        df.loc[va, "pred"] = float(df.loc[tr, LABEL_COL].mean())

    save_oof(
        "baseline.prior", df[["response_id", "session_id", LABEL_COL, "pred"]], subsample=subsample
    )
    res = score_frame(df, "baseline.prior", subsample=subsample)
    log.info("baseline.prior: logloss=%.5f (floor)", res["logloss"])
    return res


@task(
    "baseline.lo_only",
    requires="cpu",
    max_tier="cpu",
    description="per-learning-objective mean only — THE BAR every real model must clear",
)
def lo_only(
    force: bool = False,
    subsample: int | None = None,
    smoothing: float = 20.0,
) -> dict[str, Any]:
    """Smoothed per-LO mean correctness, fit within folds, using no transcript data.

    ``smoothing`` is the additive prior strength m in the empirical-Bayes shrinkage
    ``(sum + m*global) / (n + m)``. Rare LOs (the measured minimum is 1 response) get
    pulled to the global rate rather than predicting 0 or 1 and exploding log loss.
    """
    df = _train_with_folds(subsample)
    if LO_COL not in df.columns:
        raise KeyError(f"{LO_COL} missing from training features")
    df = df.copy()
    df["pred"] = np.nan

    for k in pbar(sorted(df["fold"].unique()), desc="baseline.lo_only folds", unit="fold"):
        tr_mask = df["fold"] != k
        va_mask = df["fold"] == k
        tr = df.loc[tr_mask]
        global_rate = float(tr[LABEL_COL].mean())
        agg = tr.groupby(LO_COL)[LABEL_COL].agg(["sum", "count"])
        smoothed = (agg["sum"] + smoothing * global_rate) / (agg["count"] + smoothing)
        mapped = df.loc[va_mask, LO_COL].map(smoothed)
        df.loc[va_mask, "pred"] = mapped.fillna(global_rate).to_numpy()

    save_oof(
        "baseline.lo_only",
        df[["response_id", "session_id", LABEL_COL, "pred"]],
        subsample=subsample,
    )
    res = score_frame(df, "baseline.lo_only", subsample=subsample)
    res["smoothing"] = smoothing
    log.info("baseline.lo_only: logloss=%.5f  <-- THE BAR", res["logloss"])
    return res
