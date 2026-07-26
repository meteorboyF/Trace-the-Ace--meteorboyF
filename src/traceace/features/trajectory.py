"""Trajectory and LO-position blocks — order and time allocation.

Two response-level blocks built in one pass, because both need the same per-session
window computation and the transcript is the expensive part to read.

**`trajectory` (``traj_``) — order sensitivity.** Every other block is an aggregate, and
aggregates are order-blind. A student who struggles early then masters the topic and one
whose fluency degrades into confusion produce *identical* means, rates and ratios — yet
they have opposite outcomes. This block is explicitly about shape over time inside the
LO-relevant window: length trends, first-vs-last third rates, where the final correction
falls, how the window ends, and run lengths of unaffirmed attempts.

**`lo_position` (``lopos_``) — time allocation.** Lessons run ~43 minutes and average 1.54
assessed objectives, so objectives **compete for a fixed budget**. This block captures
where an objective sits in the lesson, what share of utterances and minutes it received,
how many other objectives it competed with, how interleaved they were, and how much lesson
remained after it. This is also a directly reportable finding about time allocation in
multi-objective tutoring, independent of any model.

Both implementations live in :mod:`traceace.packaging.inference_lib` so the training and
submission paths are the same code by construction (ADR-007).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..cache import load_or_compute
from ..io import load_train_features
from ..logging_utils import get_logger
from ..packaging.inference_lib import lo_position_features, trajectory_features
from ..packaging.inference_lib import topk_spans as _shared_topk_spans
from ..paths import transcripts_dir
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, normalize_frame
from .feedback import _topk_window_frame
from .lo_alignment import TOPK, _window_texts, _windows, fit_lo_vectorizer

log = get_logger("features.trajectory")

VERSION = "v1"

# Cache key includes a hash of the code that computes this block, so editing the
# computation invalidates the cache automatically (see common.source_digest).
_SRC: str | None = None


# Window selection comes from inference_lib so training and submission agree exactly.


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
    "features.trajectory",
    requires="cpu",
    max_tier="cpu",
    description="order-sensitive within-window trends + LO time-allocation features",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    topk: int = TOPK,
) -> dict[str, Any]:
    stage_local()
    path = block_cache_path("trajectory", VERSION, subsample, source_hash=_source())
    vec = fit_lo_vectorizer()

    def compute() -> pd.DataFrame:
        feats = load_train_features()
        if subsample is not None:
            sessions = feats["session_id"].drop_duplicates().head(max(1, subsample))
            feats = feats[feats["session_id"].isin(sessions)]

        rows: list[dict[str, Any]] = []
        tdir = transcripts_dir()
        for sid, grp in pbar(
            list(feats.groupby("session_id")), desc="features.trajectory", unit="session"
        ):
            fp = tdir / f"{sid}.csv"
            if not fp.is_file():
                continue
            try:
                df = normalize_frame(pd.read_csv(fp, dtype=str))
            except Exception as exc:
                log.debug("unreadable %s (%s)", fp.name, exc)
                continue

            spans = _windows(df)
            wm = vec.transform(_window_texts(df, spans)) if spans else None
            t_seconds = df["t_seconds"].to_numpy(dtype=float)

            # Every objective's span set is needed to compute ordinal position and
            # competition, so resolve them all before emitting any row.
            lo_spans: dict[str, list[tuple[int, int]]] = {}
            for _, r in grp.iterrows():
                lo_spans[str(r["response_id"])] = _shared_topk_spans(
                    str(r["learning_objective"]), vec, wm, spans, topk
                )
            all_spans = list(lo_spans.values())

            for _, r in grp.iterrows():
                rid = str(r["response_id"])
                sub = _topk_window_frame(
                    df, str(r["learning_objective"]), vec, wm, spans, topk=topk
                )
                f: dict[str, Any] = dict(trajectory_features(sub))
                f.update(lo_position_features(lo_spans[rid], all_spans, len(df), t_seconds))
                f["response_id"] = r["response_id"]
                f["session_id"] = sid
                rows.append(f)
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.trajectory")
    return {
        "output_path": str(path),
        "n_responses": int(len(out)),
        "n_features": int(out.shape[1] - 2),
    }
