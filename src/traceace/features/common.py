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
from typing import Any

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


def source_digest(*modules: Any) -> str:
    """Short hash of the given modules' source code.

    Feature caches were keyed on a hand-written ``VERSION = "v1"`` that nobody remembered to
    bump. Editing a feature computation therefore left the stale parquet in place and
    ``load_or_compute`` served it silently — a model trained on features that no longer match
    the code that claims to produce them. Hashing the source makes invalidation automatic.

    Deliberately hashes the whole module: a docstring edit needlessly rebuilds a block
    (~2 min), which is a much cheaper mistake than serving stale features.
    """
    import hashlib
    import inspect

    h = hashlib.sha256()
    for mod in modules:
        try:
            h.update(inspect.getsource(mod).encode("utf-8"))
        except (OSError, TypeError):  # built-in / interactive — cannot hash
            h.update(repr(mod).encode("utf-8"))
    return h.hexdigest()[:10]


def block_cache_path(
    block: str,
    version: str,
    subsample: int | None,
    source_hash: str | None = None,
) -> Path:
    """Cache path for a feature block.

    ``source_hash`` should be a :func:`source_digest` of the modules that compute the block,
    so editing the computation invalidates the cache automatically instead of silently
    reusing features the current code would no longer produce.
    """
    from ..paths import features_dir

    suffix = "" if subsample is None else f"_sub{subsample}"
    tag = version if source_hash is None else f"{version}_{source_hash}"
    return features_dir() / f"{block}{suffix}__{tag}.parquet"
