"""Frozen transcript embeddings — extracted ONCE on L4, cached forever.

Measured token distribution (docs/DATA.md): median 5.3K tokens/session, p99 8.1K,
max 11.5K. A ModernBERT-base context of 8192 tokens therefore covers ~99% of sessions
in a single window, and the whole test set encodes in ~6 minutes on an A100 — the 6-hour
cap is not a binding constraint.

**Cost discipline (§4).** This is the only routine GPU task in the cheap-first ladder.
The cache check is unconditional and loud: a hit skips extraction entirely. Re-running
this task should be a no-op forever after the first successful run. Always smoke it with
``subsample=500`` before the full extraction.

Embeddings are stored per **session** (mean-pooled over chunks) *and* per **window**,
because :mod:`traceace.features.lo_alignment` needs window-level vectors to localize
key moments for the embedding backend.

Model: ``answerdotai/ModernBERT-base`` — Apache-2.0, commercial use permitted
(docs/EXTERNAL_ASSETS.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..cache import is_cached
from ..config import get_config
from ..io import write_parquet
from ..logging_utils import get_logger
from ..progress import heartbeat, pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, iter_session_frames

log = get_logger("features.embeddings")

VERSION = "v1"
PREFIX = "emb_"


def _load_model(model_name: str):
    """Load a sentence-transformers model onto the best available device."""
    from sentence_transformers import SentenceTransformer

    with heartbeat(f"loading {model_name}"):
        model = SentenceTransformer(model_name)
    return model


def _session_text(df: pd.DataFrame, max_chars: int) -> str:
    """Render a transcript to text with role tags, truncated to a char budget.

    Role tags are included because *who* said something is the signal; ``background``
    is kept (it contains real, misattributed tutor speech) but tagged distinctly so the
    encoder can learn to discount it.
    """
    parts = [f"{r}: {c}" for r, c in zip(df["role"].tolist(), df["content"].tolist())]
    text = "\n".join(parts)
    return text[:max_chars]


@task(
    "features.embeddings",
    requires="l4",
    max_tier="a100",
    description="frozen transcript embeddings — extract ONCE on L4, cache forever",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Extract mean-pooled session embeddings and cache them to parquet.

    Requires at least an L4 (``requires="l4"``); the tier guard refuses to run this on
    CPU and refuses tiers above A100 without ``allow_waste=True``.
    """
    cfg = get_config()
    model_name = model_name or str(cfg.get("embeddings", "model_name"))
    batch_size = int(batch_size or cfg.get("embeddings", "batch_size", default=32))
    max_tokens = int(cfg.get("embeddings", "max_tokens", default=8192))
    cache_version = str(cfg.get("embeddings", "cache_version", default="v1"))
    max_chars = max_tokens * 4  # measured ~3.45 chars/token; 4 is a safe over-estimate

    tag = f"{model_name.split('/')[-1]}_{cache_version}"
    path = block_cache_path("embeddings", tag, subsample)

    # LOUD, UNCONDITIONAL cache check — this is the expensive task.
    if is_cached(path) and not force:
        log.warning(
            "CACHE HIT for embeddings (%s). SKIPPING GPU extraction entirely. "
            "Pass force=True only if the model or config changed.",
            path.name,
        )
        df = pd.read_parquet(path, columns=["session_id"])
        return {
            "output_path": str(path),
            "n_sessions": int(len(df)),
            "model": model_name,
            "cached": True,
        }

    stage_local()
    model = _load_model(model_name)

    sids: list[str] = []
    texts: list[str] = []
    for sid, df in pbar(
        iter_session_frames(subsample=subsample), desc="collect transcripts", unit="session"
    ):
        sids.append(sid)
        texts.append(_session_text(df, max_chars))

    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,  # we drive our own bar; never trust library bars in submission
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs, dtype=np.float32)

    out = pd.DataFrame(vecs, columns=[f"{PREFIX}{i:03d}" for i in range(vecs.shape[1])])
    out.insert(0, "session_id", sids)
    write_parquet(out, path)
    log.info("embeddings: %d sessions x %d dims -> %s", len(out), vecs.shape[1], path)
    return {
        "output_path": str(path),
        "n_sessions": int(len(out)),
        "dim": int(vecs.shape[1]),
        "model": model_name,
        "cached": False,
    }
