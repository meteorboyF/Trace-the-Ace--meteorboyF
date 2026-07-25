"""Content block — *what was said*, as reduced dense vectors.

**The design point.** Collapsing window embeddings to a single cosine scalar (as
``lo_alignment`` does) throws away almost everything the encoder computed: it answers
"was this topic discussed here?" but not "what was actually said about it?". Those are
orthogonal kinds of evidence. Talk ratio, feedback ratios and disfluency all describe the
*form* of the interaction; the pooled embedding describes its *content*.

So this block pools the top-k LO-relevant window vectors and reduces them with PCA to a
modest number of components, which go into the GBDT as ordinary numeric features. PCA
rather than raw 384 dims because:

* trees split on one feature at a time and handle a few dense informative axes far better
  than hundreds of thin correlated ones;
* it keeps the block comparable in width to the others, so the ablation is a fair test;
* it keeps the submission asset small.

**Fitted on training data only.** The PCA basis is estimated from training windows and
persisted, so inference is a pure transform — no test-set fitting, which the rules require
and which also keeps the block deterministic.

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
from ..progress import heartbeat, pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, normalize_frame
from .lo_alignment import TOPK, _windows

log = get_logger("features.content")

VERSION = "v1"
PREFIX = "cont_"
DEFAULT_COMPONENTS = 48


def pca_path(tag: str, n_components: int) -> Path:
    return interim_dir() / f"content_pca_{tag}_k{n_components}.joblib"


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
    import joblib
    from sklearn.decomposition import PCA

    from .window_embeddings import DEFAULT_MODEL, lo_embedding_path
    from .window_embeddings import VERSION as WE_VERSION

    cfg = get_config()
    model_name = str(cfg.get("embeddings", "alignment_model", default=DEFAULT_MODEL))
    tag = f"{model_name.split('/')[-1]}_{WE_VERSION}"
    win_path = block_cache_path("window_embeddings", tag, subsample)
    lo_path = lo_embedding_path(tag if subsample is None else f"{tag}_sub{subsample}")

    if not win_path.is_file() or not lo_path.is_file():
        raise FileNotFoundError(
            f"window embeddings missing ({win_path.name}). Run "
            "tasks.run('features.window_embeddings') on an L4 first (smoke subsample=500)."
        )

    path = block_cache_path("content", f"{VERSION}_k{n_components}", subsample)

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
        k = int(min(n_components, X.shape[1], max(X.shape[0] - 1, 1)))

        # PCA basis is fit on TRAINING pooled vectors only, then persisted for inference.
        pp = pca_path(tag, k)
        with heartbeat(f"PCA fit ({X.shape[0]} x {X.shape[1]} -> {k})"):
            pca = PCA(n_components=k, random_state=cfg.seed)
            Z = pca.fit_transform(X)
        pp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pca": pca, "n_components": k, "model": model_name}, pp)
        log.info(
            "content PCA: %d -> %d dims, explained variance %.3f -> %s",
            X.shape[1],
            k,
            float(pca.explained_variance_ratio_.sum()),
            pp,
        )

        out = pd.DataFrame(Z, columns=[f"{PREFIX}{i:02d}" for i in range(k)])
        out.insert(0, "session_id", [s for _, s in ids])
        out.insert(0, "response_id", [r for r, _ in ids])
        return out

    out = load_or_compute(path, compute, force=force, label="features.content")
    return {
        "output_path": str(path),
        "n_responses": int(len(out)),
        "n_features": int(out.shape[1] - 2),
        "n_components": n_components,
        "model": model_name,
    }
