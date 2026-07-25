"""Feedback block — tutor corrective feedback around student attempts.

**The hypothesis.** Mastery signal lives less in *how much* a student talks than in
**how the tutor responds to what the student says**. A student who needs three attempts
and two corrections before the tutor affirms has probably not mastered the topic; one
affirmed on the first attempt probably has. Crucially this is a property of the
*interaction*, which is exactly what a transcript uniquely provides and what a
topic-difficulty lookup table cannot.

Computed at two scopes, because both are cheap and they answer different questions:

* ``fbs_`` — over the **whole session**: what is this tutor's feedback style?
* ``fb_``  — over the **LO-relevant windows only**: how did the tutor respond on *this
  topic*? This scope is response-level and therefore differs between two learning
  objectives in the same session, which is what breaks the within-session tie
  (see docs/DATA.md and ADR-003).

The implementation lives in :mod:`traceace.packaging.inference_lib` and is imported here
rather than duplicated, so the training and submission paths are the *same code* by
construction — no parity test can drift because there is only one implementation
(ADR-007).

Deliberately lexical and auditable to start with: an education researcher can read the
marker lists. The move classifier may supersede it later; the transparent version stays
valuable for the write-up regardless.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..cache import load_or_compute
from ..io import load_train_features
from ..logging_utils import get_logger
from ..packaging.inference_lib import feedback_features
from ..paths import transcripts_dir
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, normalize_frame
from .lo_alignment import TOPK, _window_texts, _windows, fit_lo_vectorizer

log = get_logger("features.feedback")

VERSION = "v1"
SESSION_PREFIX = "fbs_"
WINDOW_PREFIX = "fb_"


def _topk_window_frame(
    df: pd.DataFrame,
    lo_text: str,
    vec: Any,
    window_matrix: Any,
    spans: list[tuple[int, int]],
    topk: int = TOPK,
) -> pd.DataFrame:
    """Concatenate the top-k LO-relevant windows, in transcript order.

    Order matters here (unlike in the LO-alignment block's bag-of-statistics): feedback
    features walk the dialogue sequentially to find attempt→correction→affirmation
    episodes, so the windows must be stitched back in chronological order.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if not spans or window_matrix is None:
        return df.iloc[0:0]
    sims = cosine_similarity(vec.transform([lo_text or ""]), window_matrix).ravel()
    top = np.argsort(-sims)[: min(topk, len(sims))]
    # merge overlapping spans, then take them in chronological order
    chosen = sorted(spans[int(i)] for i in top)
    keep: list[tuple[int, int]] = []
    for s, e in chosen:
        if keep and s <= keep[-1][1]:
            keep[-1] = (keep[-1][0], max(keep[-1][1], e))
        else:
            keep.append((s, e))
    parts = [df.iloc[s:e] for s, e in keep]
    return pd.concat(parts) if parts else df.iloc[0:0]


@task(
    "features.feedback",
    requires="cpu",
    max_tier="cpu",
    description="tutor corrective feedback: attempts before affirmation, repair, correction position",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    topk: int = TOPK,
) -> dict[str, Any]:
    """Response-level feedback features (session-scope + LO-window scope)."""
    stage_local()
    path = block_cache_path("feedback", VERSION, subsample)
    vec = fit_lo_vectorizer()

    def compute() -> pd.DataFrame:
        feats = load_train_features()
        if subsample is not None:
            sessions = feats["session_id"].drop_duplicates().head(max(1, subsample))
            feats = feats[feats["session_id"].isin(sessions)]

        rows: list[dict[str, Any]] = []
        tdir = transcripts_dir()
        for sid, grp in pbar(
            list(feats.groupby("session_id")), desc="features.feedback", unit="session"
        ):
            fp = tdir / f"{sid}.csv"
            if not fp.is_file():
                continue
            try:
                df = normalize_frame(pd.read_csv(fp, dtype=str))
            except Exception as exc:
                log.debug("unreadable %s (%s)", fp.name, exc)
                continue

            # session-scope features are identical for every response in the session,
            # so compute them once
            session_feats = feedback_features(df, prefix=SESSION_PREFIX)

            spans = _windows(df)
            wm = vec.transform(_window_texts(df, spans)) if spans else None

            for _, r in grp.iterrows():
                f: dict[str, Any] = dict(session_feats)
                sub = _topk_window_frame(
                    df, str(r["learning_objective"]), vec, wm, spans, topk=topk
                )
                f.update(feedback_features(sub, prefix=WINDOW_PREFIX))
                f["response_id"] = r["response_id"]
                f["session_id"] = sid
                rows.append(f)
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.feedback")
    return {
        "output_path": str(path),
        "n_responses": int(len(out)),
        "n_features": int(out.shape[1] - 2),
    }
