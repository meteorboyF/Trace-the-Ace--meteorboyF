"""Structural features — who talks, how much, in what shape.

Cheap, fully interpretable, CPU-only. These describe the *shape* of the dialogue
rather than its content: turn counts, talk ratio, utterance length statistics, turn
alternation, and the volume of the ``background`` (diarization-failure) role.

Interpretability note for the write-up: the student talk ratio is a classic
tutoring-quality proxy in the education literature, so it is deliberately a headline
feature rather than an anonymous column in an embedding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..cache import load_or_compute
from ..logging_utils import get_logger
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import ROLES, block_cache_path, iter_session_frames, robust_stats, safe_div

log = get_logger("features.structural")

VERSION = "v1"

# Cache key includes a hash of the code that computes this block, so editing the
# computation invalidates the cache automatically (see common.source_digest).
_SRC: str | None = None
PREFIX = "struct_"


def session_structural_features(sid: str, df: pd.DataFrame) -> dict[str, Any]:
    role = df["role"]
    content = df["content"]
    lens = content.str.len().to_numpy(dtype=float)
    n = len(df)

    feats: dict[str, Any] = {"session_id": sid, f"{PREFIX}n_utterances": float(n)}

    for r in ROLES:
        m = (role == r).to_numpy()
        cnt = int(m.sum())
        chars = float(lens[m].sum()) if cnt else 0.0
        feats[f"{PREFIX}n_{r}"] = float(cnt)
        feats[f"{PREFIX}chars_{r}"] = chars
        feats[f"{PREFIX}frac_utt_{r}"] = safe_div(cnt, n)
        feats[f"{PREFIX}mean_len_{r}"] = safe_div(chars, cnt)
        if cnt:
            feats.update(robust_stats(lens[m], f"{PREFIX}len_{r}"))
        else:
            feats.update(robust_stats(np.array([]), f"{PREFIX}len_{r}"))

    total_chars = float(lens.sum())
    stu_chars = feats[f"{PREFIX}chars_student"]
    tut_chars = feats[f"{PREFIX}chars_tutor"]

    # Talk ratio: the canonical tutoring-quality proxy. Both by characters and by turns.
    feats[f"{PREFIX}student_talk_ratio"] = safe_div(stu_chars, stu_chars + tut_chars)
    feats[f"{PREFIX}student_turn_ratio"] = safe_div(
        feats[f"{PREFIX}n_student"], feats[f"{PREFIX}n_student"] + feats[f"{PREFIX}n_tutor"]
    )
    feats[f"{PREFIX}total_chars"] = total_chars

    # Turn-taking dynamics: how often does the speaker actually change?
    # Done in plain numpy on purpose: pandas' shift() introduces NA (ambiguous truth
    # value) and pandas 3's pyarrow booleans lack a cumsum kernel. This form is correct
    # and identical on pandas 2.x and 3.x.
    role_arr = role.fillna("").astype(str).to_numpy()
    neq = np.ones(n, dtype=bool)
    if n > 1:
        neq[1:] = role_arr[1:] != role_arr[:-1]
    changes = int(neq[1:].sum()) if n > 1 else 0
    feats[f"{PREFIX}role_switches"] = float(changes)
    feats[f"{PREFIX}switch_rate"] = safe_div(changes, n)

    # Consecutive-run lengths: long tutor monologues vs. rapid back-and-forth.
    run_ids = np.cumsum(neq)
    run_len = pd.Series(run_ids).groupby(run_ids).size().to_numpy(dtype=float)
    feats.update(robust_stats(run_len, f"{PREFIX}runlen"))

    # background = ASR diarization failure. Its volume is a data-quality signal.
    feats[f"{PREFIX}background_char_frac"] = safe_div(
        feats[f"{PREFIX}chars_background"], total_chars
    )
    return feats


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
    "features.structural",
    requires="cpu",
    max_tier="cpu",
    description="turn counts, talk ratio, utterance length stats, turn-taking dynamics",
)
def build(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    stage_local()
    path = block_cache_path("structural", VERSION, subsample, source_hash=_source())

    def compute() -> pd.DataFrame:
        rows = []
        it = iter_session_frames(subsample=subsample)
        for sid, df in pbar(it, desc="features.structural", unit="session"):
            rows.append(session_structural_features(sid, df))
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.structural")
    return {
        "output_path": str(path),
        "n_sessions": int(len(out)),
        "n_features": int(out.shape[1] - 1),
        "columns": [c for c in out.columns if c != "session_id"][:40],
    }
