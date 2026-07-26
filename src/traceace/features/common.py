"""Shared helpers for feature blocks.

All feature blocks follow the same contract:

* They compute **session-level** features keyed by ``session_id``, cached to parquet.
* They are joined onto response rows downstream. Because up to 10 responses share one
  session (58.8% of responses live in multi-response sessions), session-level features
  alone **cannot** separate rows within a session — that is what
  :mod:`traceace.features.lo_alignment` is for. See docs/DATA.md.
* Feature names are prefixed by block (``struct_``, ``ling_``, ``temp_``, ``lo_``) so
  ablation studies can drop a whole block by prefix.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..paths import iter_files, transcripts_dir

log = get_logger("features")

# Roles as they actually appear (three-valued; `background` is a diarization-failure
# bucket containing real speech, not pure noise — see docs/DATA.md).
ROLES = ("tutor", "student", "background")


def iter_session_frames(
    subsample: int | None = None,
    session_ids: set[str] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(session_id, transcript_df)`` reading per-session CSVs from local disk.

    Streaming keeps memory flat over 22.8k sessions. The Drive guard in
    :mod:`traceace.paths` ensures this never iterates a FUSE mount.
    """
    files = list(iter_files(transcripts_dir(), "*.csv"))
    if session_ids is None and subsample is not None:
        # All subsampled tasks must use the same cohort as cv.build and the
        # response-level blocks. Taking the first N sorted transcript filenames selects
        # an almost-disjoint cohort when train_features.csv is in a different order.
        from ..io import subsample_session_ids

        session_ids = set(subsample_session_ids(subsample))
        subsample = None
    if session_ids is not None:
        files = [f for f in files if f.stem in session_ids]
    if subsample is not None:
        files = files[:subsample]
    for path in files:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:
            log.debug("unreadable transcript %s (%s)", path.name, exc)
            continue
        yield path.stem, normalize_frame(df)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dtypes/columns of one transcript frame."""
    for col in ("session_id", "utterance_id", "role", "content", "timestamp"):
        if col not in df.columns:
            df[col] = pd.NA
    df["role"] = df["role"].astype("string").fillna("unknown").str.strip().str.lower()
    df["content"] = df["content"].astype("string").fillna("")
    df["utterance_idx"] = pd.to_numeric(df["utterance_id"], errors="coerce")
    from ..data import parse_elapsed_seconds

    df["t_seconds"] = df["timestamp"].map(parse_elapsed_seconds)
    return df.sort_values(["utterance_idx", "t_seconds"], kind="stable").reset_index(drop=True)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b else default


def robust_stats(x: np.ndarray, prefix: str, trim: float = 0.1) -> dict[str, float]:
    """Median / trimmed-mean / IQR summary.

    We prefer robust statistics for anything derived from ASR timing: automatic
    segmentation creates spurious long gaps (silence, mis-splits), and a plain mean is
    dominated by those artifacts. See docs/DECISIONS.md.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_median": 0.0,
            f"{prefix}_trimmean": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_p90": 0.0,
        }
    lo, hi = np.percentile(x, [trim * 100, (1 - trim) * 100])
    trimmed = x[(x >= lo) & (x <= hi)]
    return {
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_trimmean": float(trimmed.mean()) if trimmed.size else float(np.median(x)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
    }


def block_cache_path(block: str, version: str, subsample: int | None) -> Path:
    from ..paths import features_dir

    suffix = "" if subsample is None else f"_sub{subsample}"
    return features_dir() / f"{block}{suffix}__{version}.parquet"
