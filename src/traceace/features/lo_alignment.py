"""LO-alignment features — "key moments", and the fix for within-session ties.

**Why this block exists (the central modeling problem).** 58.8% of training responses
come from sessions with more than one response, and 38.3% of those sessions carry
*mixed* labels. Measured variance decomposition: **43.3% of total label variance is
within-session** (docs/DATA.md). Every session-level feature — structural, linguistic,
temporal, and a whole-session embedding alike — assigns *identical* values to every
response in a session, and therefore **cannot separate them even in principle**. An
oracle that knew each session's true mean still cannot beat that within-session floor.

The only way to recover that 43% is to condition features on the **learning objective**.
So for each ``(session, learning_objective)`` pair we:

1. Slide a window over the session's utterances.
2. Score each window's relevance to the LO text (cosine similarity).
3. Pool features over the **top-k most relevant windows** — including *where* in the
   session they sit, and what the dialogue looks like *inside* them.

This makes "key moments" a **modeling requirement**, not just a research angle: the
top-k windows are literally the moments the model reads, and their positions are
directly reportable as the key-moments analysis for the write-up.

**Two backends.**

* ``lexical`` (default, CPU, zero units): TF-IDF cosine. The vectorizer is fit on
  **training LO texts only**, so it is deterministic and never touches test data —
  satisfying the rule that training with absent test data must yield identical
  parameters.
* ``embedding`` (L4): dense cosine using cached window embeddings, when available.

The lexical backend runs everywhere, including the CPU self-test, so the pipeline is
never blocked on a GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..cache import load_or_compute
from ..io import load_train_features
from ..logging_utils import get_logger
from ..paths import interim_dir, transcripts_dir
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, normalize_frame

log = get_logger("features.lo_alignment")

VERSION = "v1"

# Cache key includes a hash of the code that computes this block, so editing the
# computation invalidates the cache automatically (see common.source_digest).
_SRC: str | None = None
PREFIX = "lo_"

WINDOW = 20  # utterances per window (~1.5 min of dialogue at measured pacing)
STRIDE = 10  # 50% overlap so a key moment is never split across a boundary
TOPK = 3  # pool over the 3 most LO-relevant windows


def vectorizer_path() -> Path:
    return interim_dir() / f"lo_tfidf_{VERSION}.joblib"


def fit_lo_vectorizer(force: bool = False) -> TfidfVectorizer:
    """Fit (or load) the TF-IDF vectorizer on **training LO texts only**.

    Fitting on training LO text alone keeps the transform deterministic and independent
    of the test set, which the competition rules require. The vocabulary is small and
    domain-specific (mathematics topic phrases), which is exactly what we want to match
    against transcript windows.
    """
    import joblib
    import sklearn

    path = vectorizer_path()
    feats = load_train_features()
    lo_texts = feats["learning_objective"].dropna().astype(str).drop_duplicates().tolist()
    recipe = {
        "lowercase": True,
        "stop_words": "english",
        "ngram_range": [1, 2],
        "min_df": 1,
        "sublinear_tf": True,
        "input_sha256": hashlib.sha256(
            json.dumps(sorted(lo_texts), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    if path.is_file() and not force:
        # Older artifacts stored the estimator alone, so re-pickling under the current
        # environment could hide the sklearn version it was actually fitted with.
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cached = joblib.load(path)
        if (
            isinstance(cached, dict)
            and cached.get("sklearn_version") == sklearn.__version__
            and cached.get("recipe") == recipe
            and isinstance(cached.get("vectorizer"), TfidfVectorizer)
        ):
            return cached["vectorizer"]
        log.warning("LO vectorizer provenance is stale or unknown; refitting")

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    vec.fit(lo_texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vec,
            "sklearn_version": sklearn.__version__,
            "recipe": recipe,
        },
        path,
    )
    log.info("fit LO tf-idf vectorizer on %d unique LO texts -> %s", len(lo_texts), path)
    return vec


def _windows(df: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    """Return (start, end) index pairs covering the session."""
    n = len(df)
    if n == 0:
        return []
    if n <= window:
        return [(0, n)]
    out = [(s, min(s + window, n)) for s in range(0, n - window + 1, stride)]
    if out[-1][1] < n:  # ensure the tail is covered
        out.append((max(0, n - window), n))
    return out


def _window_texts(df: pd.DataFrame, spans: list[tuple[int, int]]) -> list[str]:
    content = df["content"].tolist()
    return [" ".join(content[s:e]) for s, e in spans]


# ---------------------------------------------------------------------------
# Feature computation is DELEGATED to inference_lib — deliberately not duplicated.
# ---------------------------------------------------------------------------
# This module previously carried its own copies of `_window_dialogue_features` and
# `alignment_features`, parallel to the ones in `packaging.inference_lib`. They happened to
# agree, but nothing enforced it: a one-line edit to either would have produced a silent
# train/serve divergence in the block that localizes every other block's window. That is
# exactly how the feature-ORDER bug reached the leaderboard (ADR-013), so the duplication is
# removed rather than tested around.
#
# `inference_lib` is the single implementation because it is the copy that ships.


def _window_dialogue_features(
    df: pd.DataFrame, spans: list[tuple[int, int]], idxs: np.ndarray
) -> dict[str, float]:
    """Delegate to the shipped implementation."""
    from ..packaging.inference_lib import _window_dialogue_features as _impl

    return _impl(df, spans, idxs)


def alignment_features(
    df: pd.DataFrame,
    lo_text: str,
    vec: TfidfVectorizer,
    window_matrix: Any = None,
    spans: list[tuple[int, int]] | None = None,
    topk: int = TOPK,
) -> dict[str, float]:
    """Features for one (session, learning objective) pair.

    Thin wrapper over ``inference_lib.lo_alignment_features`` so training and the submission
    run the *same* code. ``window_matrix`` and ``spans`` are precomputed per session and
    reused across that session's objectives.
    """
    from ..packaging.inference_lib import lo_alignment_features as _impl

    if spans is None:
        spans = _windows(df)
    if not spans:
        return {}
    if window_matrix is None:
        window_matrix = vec.transform(_window_texts(df, spans))
    return _impl(df, lo_text, vec, window_matrix, spans, topk=topk)


def _gini(x: np.ndarray) -> float:
    """Concentration of relevance across windows (0=spread evenly, 1=one window)."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    if x.sum() <= 0 or x.size < 2:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


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
    "features.lo_alignment",
    requires="cpu",
    max_tier="cpu",
    description="LO-conditioned key-moment features (response-level, breaks within-session ties)",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    backend: str = "lexical",
    topk: int = TOPK,
) -> dict[str, Any]:
    """Build response-level LO-alignment features.

    Unlike the other blocks this is keyed by ``response_id``, because its whole purpose
    is to differ between responses that share a session.

    ``backend="lexical"`` scores window relevance with TF-IDF cosine (CPU, zero units).
    ``backend="embedding"`` uses cached dense vectors from ``features.window_embeddings``
    — semantically far stronger, because tutoring dialogue rarely repeats the objective's
    literal wording.
    """
    if backend == "embedding":
        return _build_embedding(force=force, subsample=subsample, topk=topk)
    if backend != "lexical":
        raise ValueError(f"unknown backend {backend!r}; use 'lexical' or 'embedding'")
    stage_local()
    path = block_cache_path(
        "lo_alignment", f"{VERSION}_{backend}", subsample, source_hash=_source()
    )
    vec = fit_lo_vectorizer()

    def compute() -> pd.DataFrame:
        feats = load_train_features()
        if subsample is not None:
            sessions = feats["session_id"].drop_duplicates().head(max(1, subsample))
            feats = feats[feats["session_id"].isin(sessions)]

        rows: list[dict[str, Any]] = []
        tdir = transcripts_dir()
        grouped = list(feats.groupby("session_id"))
        for sid, grp in pbar(grouped, desc="features.lo_alignment", unit="session"):
            fp = tdir / f"{sid}.csv"
            if not fp.is_file():
                continue
            try:
                df = normalize_frame(pd.read_csv(fp, dtype=str))
            except Exception as exc:
                log.debug("unreadable %s (%s)", fp.name, exc)
                continue
            spans = _windows(df)
            if not spans:
                continue
            # vectorize this session's windows ONCE, reuse across its LOs
            wm = vec.transform(_window_texts(df, spans))
            for _, r in grp.iterrows():
                f = alignment_features(
                    df,
                    str(r["learning_objective"]),
                    vec,
                    window_matrix=wm,
                    spans=spans,
                    topk=topk,
                )
                f["response_id"] = r["response_id"]
                f["session_id"] = sid
                rows.append(f)
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.lo_alignment")
    return {
        "output_path": str(path),
        "n_responses": int(len(out)),
        "n_features": int(out.shape[1] - 2),
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Semantic (embedding) backend
# ---------------------------------------------------------------------------
def _build_embedding(
    force: bool = False, subsample: int | None = None, topk: int = TOPK
) -> dict[str, Any]:
    """LO-alignment scored by dense cosine instead of lexical overlap.

    Reads the cached window vectors from ``features.window_embeddings`` (extracted once
    on GPU) and the learning-objective vectors, then computes exactly the same feature
    set as the lexical backend so the two are directly comparable in an ablation — the
    only thing that changes is *how relevance is measured*.
    """
    from ..config import get_config
    from .window_embeddings import DEFAULT_MODEL, lo_embedding_path, window_embedding_path

    cfg = get_config()
    model_name = str(cfg.get("embeddings", "alignment_model", default=DEFAULT_MODEL))
    win_path = window_embedding_path(model_name, subsample)
    lo_path = lo_embedding_path(model_name, subsample)

    if not win_path.is_file() or not lo_path.is_file():
        raise FileNotFoundError(
            f"window embeddings missing ({win_path.name} / {lo_path.name}). Run "
            "tasks.run('features.window_embeddings') on an L4 first "
            "(smoke with subsample=500)."
        )

    path = block_cache_path(
        "lo_alignment", f"{VERSION}_embedding", subsample, source_hash=_source()
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
        # group window vectors by session once
        win_sorted = win.sort_values(["session_id", "window_idx"])
        win_groups = {
            sid: (g[edim].to_numpy(dtype=np.float32), g["centre_pos"].to_numpy(dtype=float))
            for sid, g in win_sorted.groupby("session_id")
        }

        feats = load_train_features()
        if subsample is not None:
            feats = feats[feats["session_id"].isin(win_groups.keys())]

        tdir = transcripts_dir()
        rows: list[dict[str, Any]] = []
        for sid, grp in pbar(
            list(feats.groupby("session_id")), desc="lo_alignment[embedding]", unit="session"
        ):
            if sid not in win_groups:
                continue
            W, pos = win_groups[sid]
            # dialogue-window features still need the transcript itself
            fp = tdir / f"{sid}.csv"
            if not fp.is_file():
                continue
            try:
                df = normalize_frame(pd.read_csv(fp, dtype=str))
            except Exception:
                continue
            spans = _windows(df)
            if len(spans) != len(W):  # window config changed since extraction
                continue

            for _, r in grp.iterrows():
                v = lo_vecs.get(str(r["learning_objective_id"]))
                if v is None:
                    continue
                # both sides are L2-normalized, so a dot product IS cosine similarity
                sims = W @ v
                f = _features_from_sims(sims, pos, df, spans, topk)
                f["response_id"] = r["response_id"]
                f["session_id"] = sid
                rows.append(f)
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.lo_alignment[embedding]")
    return {
        "output_path": str(path),
        "n_responses": int(len(out)),
        "n_features": int(out.shape[1] - 2),
        "backend": "embedding",
        "model": model_name,
    }


def _features_from_sims(
    sims: np.ndarray,
    pos: np.ndarray,
    df: pd.DataFrame,
    spans: list[tuple[int, int]],
    topk: int,
) -> dict[str, float]:
    """Identical feature set to the lexical backend, given precomputed similarities.

    Keeping one feature definition for both backends is what makes the lexical-vs-semantic
    ablation a clean comparison rather than a confounded one.
    """
    n_win = len(sims)
    order = np.argsort(-sims)
    top = order[: min(topk, n_win)]
    best = int(order[0])

    feats: dict[str, float] = {
        f"{PREFIX}sim_max": float(sims[best]),
        f"{PREFIX}sim_topk_mean": float(sims[top].mean()),
        f"{PREFIX}sim_mean": float(sims.mean()),
        f"{PREFIX}sim_std": float(sims.std()),
        f"{PREFIX}sim_sum": float(sims.sum()),
        f"{PREFIX}sim_frac_above_half_max": float(
            np.mean(sims >= 0.5 * sims[best]) if sims[best] > 0 else 0.0
        ),
        f"{PREFIX}best_pos": float(pos[best]),
        f"{PREFIX}topk_pos_mean": float(pos[top].mean()),
        f"{PREFIX}topk_pos_spread": (
            float(pos[top].max() - pos[top].min()) if len(top) > 1 else 0.0
        ),
        f"{PREFIX}n_windows": float(n_win),
        f"{PREFIX}sim_gini": _gini(sims),
    }
    feats.update(_window_dialogue_features(df, spans, top))
    return feats
