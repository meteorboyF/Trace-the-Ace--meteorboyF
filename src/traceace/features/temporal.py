"""Temporal features — latency, pacing, and how they drift across a session.

Timestamps are **relative elapsed** ``H:MM:SS`` (docs/DATA.md), so all features are
durations, not wall-clock. Sessions are tightly clustered around ~43 minutes, so pacing
comparisons across sessions are meaningful.

**Robust statistics throughout.** ASR segmentation creates spurious gaps: a silence, a
mis-split utterance, or a dropped connection produces outlier latencies that would
dominate a plain mean. We therefore report median / trimmed-mean / IQR (see
``common.robust_stats``) rather than mean/std. This is an explicit decision recorded in
docs/DECISIONS.md.

The pedagogically interesting quantity is **student response latency** — how long the
student takes to answer after a tutor turn. Long or lengthening latency is a candidate
struggle signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..cache import load_or_compute
from ..logging_utils import get_logger
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, iter_session_frames, robust_stats, safe_div

log = get_logger("features.temporal")

VERSION = "v1"
PREFIX = "temp_"

# Gaps longer than this are almost certainly breaks/technical issues, not thinking time.
_MAX_PLAUSIBLE_GAP_S = 120.0


def _slope(y: np.ndarray) -> float:
    """Normalized least-squares slope of y over its index (0..1)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        return 0.0
    x = np.linspace(0.0, 1.0, y.size)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    return float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0


def session_temporal_features(sid: str, df: pd.DataFrame) -> dict[str, Any]:
    feats: dict[str, Any] = {"session_id": sid}
    t = df["t_seconds"].to_numpy(dtype=float)
    role = df["role"].to_numpy()

    finite = np.isfinite(t)
    duration = float(np.nanmax(t) - np.nanmin(t)) if finite.any() else 0.0
    feats[f"{PREFIX}duration_s"] = duration
    feats[f"{PREFIX}frac_timestamped"] = safe_div(int(finite.sum()), len(df))

    # inter-utterance gaps
    gaps = np.diff(t)
    gaps = gaps[np.isfinite(gaps)]
    gaps = gaps[(gaps >= 0) & (gaps <= _MAX_PLAUSIBLE_GAP_S)]
    feats.update(robust_stats(gaps, f"{PREFIX}gap"))
    feats[f"{PREFIX}n_long_pauses"] = float(np.sum(gaps > 10.0))
    feats[f"{PREFIX}long_pause_rate"] = safe_div(float(np.sum(gaps > 10.0)), gaps.size)

    # student response latency: tutor turn -> next student turn
    lat = []
    for i in range(len(df) - 1):
        if role[i] == "tutor" and role[i + 1] == "student":
            d = t[i + 1] - t[i]
            if np.isfinite(d) and 0 <= d <= _MAX_PLAUSIBLE_GAP_S:
                lat.append(d)
    lat_arr = np.array(lat, dtype=float)
    feats.update(robust_stats(lat_arr, f"{PREFIX}student_latency"))
    feats[f"{PREFIX}n_student_responses"] = float(lat_arr.size)
    # Does the student get slower as the session progresses? (fatigue / difficulty)
    feats[f"{PREFIX}student_latency_slope"] = _slope(lat_arr)

    # pacing: utterances per minute, and how it drifts
    feats[f"{PREFIX}utt_per_min"] = safe_div(len(df), duration / 60.0)
    if finite.sum() > 10 and duration > 0:
        # utterances in each third of the session
        tt = t[finite]
        lo, hi = float(np.nanmin(tt)), float(np.nanmax(tt))
        edges = [lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi]
        thirds = [float(np.sum((tt >= edges[i]) & (tt < edges[i + 1]))) for i in range(3)]
        total = max(sum(thirds), 1.0)
        for i, v in enumerate(thirds):
            feats[f"{PREFIX}utt_frac_third{i + 1}"] = v / total
    else:
        for i in range(3):
            feats[f"{PREFIX}utt_frac_third{i + 1}"] = 0.0

    return feats


@task(
    "features.temporal",
    requires="cpu",
    max_tier="cpu",
    description="inter-utterance latency and pacing trends (robust statistics)",
)
def build(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    stage_local()
    path = block_cache_path("temporal", VERSION, subsample)

    def compute() -> pd.DataFrame:
        rows = []
        it = iter_session_frames(subsample=subsample)
        for sid, df in pbar(it, desc="features.temporal", unit="session"):
            rows.append(session_temporal_features(sid, df))
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.temporal")
    return {
        "output_path": str(path),
        "n_sessions": int(len(out)),
        "n_features": int(out.shape[1] - 1),
    }
