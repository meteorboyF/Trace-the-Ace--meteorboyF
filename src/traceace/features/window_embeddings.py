"""Window-level embeddings for **semantic** LO-alignment.

The lexical (TF-IDF) alignment backend can only match a learning objective to a window
when they share literal words. Tutoring dialogue rarely does: an objective phrased
"multiply a 2-digit number by a 1-digit number" is discussed as *"okay so we've got
twenty-seven lots of three, what's seven threes?"* — near-zero lexical overlap, near-total
semantic overlap. That mismatch caps how well the LO-alignment block can localize the
topic, which matters because it is the block carrying essentially all of the model's
transcript signal.

This task embeds every sliding window once on GPU and caches the vectors forever, so the
alignment backend becomes a cheap cosine lookup.

**Model choice.** ``BAAI/bge-small-en-v1.5`` — **MIT licensed** (commercial use fine),
33M parameters, 512-token context, and purpose-built for *asymmetric retrieval*: short
query against longer passage, which is exactly the (objective text → dialogue window)
shape here. Small enough that the full corpus embeds in well under an hour on an L4.
See docs/EXTERNAL_ASSETS.md.

**Cost discipline.** Measured corpus ≈ 22.8K sessions × ~26 windows ≈ 590K windows at
~400 tokens each ≈ 240M tokens. Always smoke with ``subsample=500`` before the full run,
and never re-run: the cache check is unconditional and loud.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import is_cached
from ..config import get_config
from ..io import load_train_features, write_parquet
from ..logging_utils import get_logger
from ..paths import interim_dir, transcripts_dir
from ..progress import heartbeat, pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, normalize_frame
from .lo_alignment import _window_texts, _windows

log = get_logger("features.window_embeddings")

VERSION = "v1"

# Cache key includes a hash of the code that computes this block, so editing the
# computation invalidates the cache automatically (see common.source_digest).
_SRC: str | None = None
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction prefix on the QUERY side only (the LO text).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def artifact_tag(model_name: str) -> str:
    """Filesystem-safe identity for the exact model, not merely its basename."""
    name = model_name.rsplit("/", 1)[-1]
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{name}_{digest}_{VERSION}"


def window_embedding_path(model_name: str, subsample: int | None) -> Path:
    return block_cache_path(
        "window_embeddings", artifact_tag(model_name), subsample, source_hash=_source()
    )


def lo_embedding_path(model_name: str, subsample: int | None) -> Path:
    suffix = "" if subsample is None else f"_sub{subsample}"
    return interim_dir() / f"lo_embeddings_{artifact_tag(model_name)}_{_source()}{suffix}.parquet"


def _load_model(model_name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    with heartbeat(f"loading {model_name}"):
        return SentenceTransformer(model_name, device=device)


def _source() -> str:
    """Digest of the code that produces this block (memoized)."""
    global _SRC
    if _SRC is None:
        import sys

        from ..packaging import inference_lib
        from .common import source_digest

        _SRC = source_digest(sys.modules[__name__], inference_lib)
    return _SRC


@task(
    "features.window_embeddings",
    requires="cpu",
    max_tier="a100",
    description="embed LO texts + sliding windows once on GPU; cache forever",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    model_name: str | None = None,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, Any]:
    """Embed every window and every learning-objective text, and cache both.

    Declared ``requires="cpu"`` so it can be *smoke-tested* on CPU without a GPU (the
    embedding model runs anywhere, just slowly). Run the full extraction on an L4.
    """
    cfg = get_config()
    model_name = model_name or str(cfg.get("embeddings", "alignment_model", default=DEFAULT_MODEL))
    win_path = window_embedding_path(model_name, subsample)
    lo_path = lo_embedding_path(model_name, subsample)

    if is_cached(win_path) and lo_path.is_file() and not force:
        log.warning(
            "CACHE HIT for window embeddings (%s). SKIPPING GPU work entirely. "
            "Pass force=True only if the model or window config changed.",
            win_path.name,
        )
        prev = pd.read_parquet(win_path, columns=["session_id"])
        return {
            "output_path": str(win_path),
            "lo_path": str(lo_path),
            "n_windows": int(len(prev)),
            "model": model_name,
            "cached": True,
        }

    stage_local()
    model = _load_model(model_name, device=device)

    # --- 1. learning-objective texts (tiny: ~398 unique) --------------------
    feats = load_train_features()
    lo_texts = (
        feats[["learning_objective_id", "learning_objective"]]
        .drop_duplicates("learning_objective_id")
        .reset_index(drop=True)
    )
    lo_vecs = model.encode(
        [QUERY_PREFIX + str(t) for t in lo_texts["learning_objective"]],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    lo_df = pd.DataFrame(lo_vecs, columns=[f"e{i:03d}" for i in range(lo_vecs.shape[1])])
    lo_df.insert(0, "learning_objective_id", lo_texts["learning_objective_id"].to_numpy())
    write_parquet(lo_df, lo_path)
    log.info("embedded %d learning objectives -> %s", len(lo_df), lo_path)

    # --- 2. sliding windows over every session ------------------------------
    sessions = feats["session_id"].drop_duplicates()
    if subsample is not None:
        sessions = sessions.head(max(1, subsample))
    tdir = transcripts_dir()

    all_texts: list[str] = []
    index: list[tuple[str, int, float]] = []  # (session_id, window_idx, centre_pos)
    for sid in pbar(sessions.tolist(), desc="collect windows", unit="session"):
        fp = tdir / f"{sid}.csv"
        if not fp.is_file():
            continue
        try:
            df = normalize_frame(pd.read_csv(fp, dtype=str))
        except Exception:
            continue
        spans = _windows(df)
        if not spans:
            continue
        texts = _window_texts(df, spans)
        n_utt = max(len(df), 1)
        for wi, ((s, e), txt) in enumerate(zip(spans, texts)):
            all_texts.append(txt)
            index.append((sid, wi, (s + e) / 2.0 / n_utt))

    if not all_texts:
        raise RuntimeError("no windows collected — did staging run?")

    log.info("encoding %d windows with %s", len(all_texts), model_name)
    vecs = model.encode(
        all_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    out = pd.DataFrame(vecs, columns=[f"e{i:03d}" for i in range(vecs.shape[1])])
    idx = pd.DataFrame(index, columns=["session_id", "window_idx", "centre_pos"])
    out = pd.concat([idx, out], axis=1)
    write_parquet(out, win_path)

    log.info("window embeddings: %d windows x %d dims -> %s", len(out), vecs.shape[1], win_path)
    return {
        "output_path": str(win_path),
        "lo_path": str(lo_path),
        "n_windows": int(len(out)),
        "n_sessions": int(out["session_id"].nunique()),
        "dim": int(vecs.shape[1]),
        "model": model_name,
        "cached": False,
    }
