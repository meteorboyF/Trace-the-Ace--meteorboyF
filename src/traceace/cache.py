"""Parquet-backed caches with loud skips.

Expensive outputs (features, embeddings, OOF predictions) are cached to parquet on
the local working disk and reused unconditionally. The cache check is **loud** (§4):
a hit prints that it is skipping recomputation, so the operator always knows whether
a paid GPU is actually doing work. ``force=True`` recomputes.

A cache key bundles a logical ``name`` and a ``version`` string so bumping the
version in ``conf/base.yaml`` (e.g. a new embedding model) invalidates the old cache
without manual deletion.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .io import read_parquet, write_parquet
from .logging_utils import get_logger
from .paths import features_dir

log = get_logger("cache")


def feature_cache_path(name: str, version: str = "v1") -> Path:
    """Standard path for a cached feature/embedding block."""
    return features_dir() / f"{name}__{version}.parquet"


def is_cached(path: Path) -> bool:
    return path.is_file()


def load_or_compute(
    path: Path,
    compute: Callable[[], pd.DataFrame],
    force: bool = False,
    label: str | None = None,
) -> pd.DataFrame:
    """Return cached parquet at ``path`` or compute+cache it.

    Parameters
    ----------
    compute:
        Zero-arg callable producing the DataFrame when the cache misses.
    force:
        Recompute even if cached.
    label:
        Human name for log lines (defaults to the file stem).
    """
    label = label or path.stem
    if is_cached(path) and not force:
        log.info("CACHE HIT  %s -> loading %s (pass force=True to recompute)", label, path)
        return read_parquet(path)
    if force and is_cached(path):
        log.info("CACHE FORCED recompute of %s (%s)", label, path)
    else:
        log.info("CACHE MISS %s -> computing", label)
    df = compute()
    write_parquet(df, path)
    log.info("CACHE WROTE %s (%d rows) -> %s", label, len(df), path)
    return df
