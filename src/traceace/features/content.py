"""Content block — *what was said*, as pooled dense vectors.

**The design point.** Collapsing window embeddings to a single cosine scalar (as
``lo_alignment`` does) throws away almost everything the encoder computed: it answers
"was this topic discussed here?" but not "what was actually said about it?". Those are
orthogonal kinds of evidence. Talk ratio, feedback ratios and disfluency all describe the
*form* of the interaction; the pooled embedding describes its *content*.

This block pools the top-k LO-relevant window vectors and stores the raw frozen-encoder
coordinates. Dimensionality reduction is deliberately deferred to model training, where
PCA is fit independently inside each outer training fold. Fitting one global PCA before
cross-validation lets validation covariates influence the representation and makes an
honest deployment comparison unnecessarily ambiguous.

* trees split on one feature at a time and handle a few dense informative axes far better
  than hundreds of thin correlated ones;
* it keeps the block comparable in width to the others, so the ablation is a fair test;
The fold-specific PCA transforms remain research artifacts until encoder inference is
implemented in the submission.

Requires ``features.window_embeddings`` (GPU, once). Without it this task fails loudly
rather than silently degrading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import load_or_compute
from ..config import get_config
from ..io import load_train_features
from ..logging_utils import get_logger
from ..paths import interim_dir, transcripts_dir
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, normalize_frame
from .lo_alignment import TOPK, _windows

log = get_logger("features.content")

VERSION = "v2"

# Cache key includes a hash of the code that computes this block, so editing the
# computation invalidates the cache automatically (see common.source_digest).
_SRC: str | None = None
PREFIX = "cont_raw_"
DEFAULT_COMPONENTS = 48


def pca_path(
    tag: str,
    n_components: int,
    topk: int,
    subsample: int | None,
    source_hash: str,
) -> Path:
    suffix = "" if subsample is None else f"_sub{subsample}"
    return interim_dir() / (
        f"content_pca_{tag}_k{n_components}_top{topk}_{source_hash}{suffix}.joblib"
    )


def _pool(vectors: np.ndarray, sims: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Similarity-weighted mean of the top-k window vectors.

    Weighting by relevance rather than taking a flat mean means a window that barely
    matches the objective contributes proportionally less to the topic's representation.
    """
    w = np.clip(sims[top], 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return (vectors[top] * w[:, None]).sum(axis=0)


def _embedding_paths(model_name: str, subsample: int | None) -> tuple[Path, Path]:
    """Resolve producer-owned embedding paths without duplicating its cache identity."""
    from .window_embeddings import lo_embedding_path, window_embedding_path

    return (
        window_embedding_path(model_name, subsample),
        lo_embedding_path(model_name, subsample),
    )


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
    "features.content",
    requires="cpu",
    max_tier="cpu",
    description="pooled top-k window embeddings, PCA-reduced — WHAT was said",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    topk: int = TOPK,
    n_components: int = DEFAULT_COMPONENTS,
) -> dict[str, Any]:
    """Response-level content vectors from cached window embeddings.

    CPU task: the GPU cost lives entirely in ``features.window_embeddings``; this is a
    pooling + PCA pass over cached vectors.
    """
    from .window_embeddings import DEFAULT_MODEL

    cfg = get_config()
    model_name = str(cfg.get("embeddings", "alignment_model", default=DEFAULT_MODEL))
    win_path, lo_path = _embedding_paths(model_name, subsample)

    if not win_path.is_file() or not lo_path.is_file():
        raise FileNotFoundError(
            f"window embeddings missing ({win_path.name}). Run "
            "tasks.run('features.window_embeddings') on an L4 first (smoke subsample=500)."
        )

    path = block_cache_path(
        "content",
        f"{VERSION}_raw_top{topk}",
        subsample,
        source_hash=_source(),
    )

    def compute() -> pd.DataFrame:
        stage_local()
        win = pd.read_parquet(win_path)
        lo = pd.read_parquet(lo_path)
        edim = [c for c in win.columns if c.startswith("e")]

        lo_vecs = {
            str(r["learning_objective_id"]): np.asarray([r[c] for c in edim], dtype=np.float32)
            for _, r in lo.iterrows()
        }
        win_sorted = win.sort_values(["session_id", "window_idx"])
        win_groups = {
            sid: g[edim].to_numpy(dtype=np.float32) for sid, g in win_sorted.groupby("session_id")
        }

        feats = load_train_features()
        if subsample is not None:
            feats = feats[feats["session_id"].isin(win_groups.keys())]

        tdir = transcripts_dir()
        ids: list[tuple[str, str]] = []
        pooled: list[np.ndarray] = []

        for sid, grp in pbar(
            list(feats.groupby("session_id")), desc="features.content pool", unit="session"
        ):
            W = win_groups.get(sid)
            if W is None:
                continue
            fp = tdir / f"{sid}.csv"
            if not fp.is_file():
                continue
            try:
                df = normalize_frame(pd.read_csv(fp, dtype=str))
            except Exception:
                continue
            if len(_windows(df)) != len(W):
                continue
            for _, r in grp.iterrows():
                v = lo_vecs.get(str(r["learning_objective_id"]))
                if v is None:
                    continue
                sims = W @ v  # both L2-normalized => cosine
                top = np.argsort(-sims)[: min(topk, len(sims))]
                pooled.append(_pool(W, sims, top))
                ids.append((str(r["response_id"]), sid))

        if not pooled:
            raise RuntimeError("no content vectors pooled — window embeddings may be stale")

        X = np.vstack(pooled).astype(np.float32)
        out = pd.DataFrame(X, columns=[f"{PREFIX}{i:03d}" for i in range(X.shape[1])])
        out.insert(0, "session_id", [s for _, s in ids])
        out.insert(0, "response_id", [r for r, _ in ids])
        log.info(
            "content raw pool: %d responses x %d dims; PCA will be fit inside each CV fold",
            len(out),
            X.shape[1],
        )
        return out

    out = load_or_compute(path, compute, force=force, label="features.content")
    return {
        "output_path": str(path),
        "n_responses": int(len(out)),
        "n_features": int(out.shape[1] - 2),
        "n_components": int(out.shape[1] - 2),
        "model": model_name,
    }
