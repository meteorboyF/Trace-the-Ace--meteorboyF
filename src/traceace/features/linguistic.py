"""Linguistic features — questioning, hedging, affirmation, and ASR disfluency.

Two families here.

**Pedagogical markers** (question density, hedging/confusion, affirmation, negation)
are lexicon-based and deliberately transparent: an education researcher can read the
lexicon and audit exactly what the model keys on, which is worth more for the write-up
than an opaque embedding dimension.

**Disfluency markers** exploit the measured fact that this corpus is *voice-transcribed*
(ASR), not typed chat (docs/DATA.md). Filler rate, false starts, self-corrections,
repetition and ``[unclear]`` density are computed **separately for student and tutor**.
Student disfluency is a candidate uncertainty marker — hesitation is a well-documented
correlate of low confidence in spoken tutoring — and it is free on CPU.

``[unclear]`` occurs in ~30% of all utterances, so its density is also an audio-quality
covariate: a session the ASR struggled with may look "quiet" for reasons that have
nothing to do with the student. That confound is called out in FINDINGS.md.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..cache import load_or_compute
from ..logging_utils import get_logger
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task
from .common import block_cache_path, iter_session_frames, robust_stats, safe_div

log = get_logger("features.linguistic")

VERSION = "v1"

# Cache key includes a hash of the code that computes this block, so editing the
# computation invalidates the cache automatically (see common.source_digest).
_SRC: str | None = None
PREFIX = "ling_"

# --- lexicons (auditable by a non-ML reader; keep them small and legible) -----
HEDGE = {
    "maybe",
    "perhaps",
    "probably",
    "guess",
    "think",
    "might",
    "possibly",
    "not sure",
    "unsure",
    "confused",
    "confusing",
    "don't know",
    "dont know",
    "no idea",
    "i don't get",
    "i dont get",
    "lost",
    "stuck",
    "hard",
    "difficult",
}
AFFIRM = {
    "yes",
    "yeah",
    "yep",
    "correct",
    "exactly",
    "right",
    "well done",
    "good job",
    "perfect",
    "brilliant",
    "great",
    "excellent",
    "nice",
    "spot on",
    "that's it",
    "lovely",
    "fantastic",
    "super",
}
NEGATE = {"no", "not quite", "incorrect", "wrong", "nope", "not really", "almost"}
FILLER = {"um", "uh", "erm", "er", "hmm", "mm", "mmm", "like", "you know", "sort of", "kind of"}
# Metacognitive / understanding checks — a named tutoring move in the literature.
UNDERSTAND_CHECK = {
    "does that make sense",
    "make sense",
    "do you understand",
    "got it",
    "is that clear",
    "are you with me",
    "any questions",
    "can you see why",
    "do you follow",
}

_WORD_RE = re.compile(r"[a-z']+")
_UNCLEAR_RE = re.compile(r"\[unclear\]", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[[^\]]{0,40}\]")
# false start: a word fragment cut off by a dash, e.g. "so we ca- can we"
_FALSE_START_RE = re.compile(r"\b[a-z]{1,12}-(?:\s|$)", re.IGNORECASE)
# self-correction cues
_SELFCORR_RE = re.compile(
    r"\b(i mean|sorry|no wait|actually|rather|let me rephrase|scratch that)\b", re.IGNORECASE
)


def _count_phrases(text_lower: str, phrases: set[str]) -> int:
    return sum(text_lower.count(p) for p in phrases)


def _repetition_rate(words: list[str]) -> float:
    """Fraction of adjacent word pairs that are identical ('the the', 'is is').

    A classic disfluency signature that survives ASR transcription.
    """
    if len(words) < 2:
        return 0.0
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return reps / (len(words) - 1)


def _role_linguistics(texts: pd.Series, role: str) -> dict[str, float]:
    """Compute the marker set for one role's utterances."""
    p = f"{PREFIX}{role}_"
    n_utt = len(texts)
    if n_utt == 0:
        keys = [
            "q_rate",
            "hedge_rate",
            "affirm_rate",
            "negate_rate",
            "filler_rate",
            "unclear_rate",
            "false_start_rate",
            "selfcorr_rate",
            "repetition_rate",
            "check_rate",
            "excl_rate",
            "mean_words",
            "type_token_ratio",
        ]
        out = {f"{p}{k}": 0.0 for k in keys}
        out.update(robust_stats(np.array([]), f"{p}words"))
        return out

    joined = "\n".join(texts.tolist())
    lower = joined.lower()
    # strip bracket tags before word statistics so [unclear] doesn't inflate counts
    stripped = _BRACKET_RE.sub(" ", lower)
    words = _WORD_RE.findall(stripped)
    n_words = max(len(words), 1)

    n_q = int(texts.str.contains(r"\?", regex=True, na=False).sum())
    n_excl = int(texts.str.contains(r"!", regex=True, na=False).sum())
    word_counts = texts.map(lambda s: len(_WORD_RE.findall(_BRACKET_RE.sub(" ", s.lower()))))

    out = {
        # per-utterance rates
        f"{p}q_rate": safe_div(n_q, n_utt),
        f"{p}excl_rate": safe_div(n_excl, n_utt),
        f"{p}unclear_rate": safe_div(len(_UNCLEAR_RE.findall(joined)), n_utt),
        f"{p}false_start_rate": safe_div(len(_FALSE_START_RE.findall(stripped)), n_utt),
        f"{p}selfcorr_rate": safe_div(len(_SELFCORR_RE.findall(lower)), n_utt),
        # per-word rates (lexicon hits)
        f"{p}hedge_rate": safe_div(_count_phrases(lower, HEDGE), n_words) * 100.0,
        f"{p}affirm_rate": safe_div(_count_phrases(lower, AFFIRM), n_words) * 100.0,
        f"{p}negate_rate": safe_div(_count_phrases(lower, NEGATE), n_words) * 100.0,
        f"{p}filler_rate": safe_div(_count_phrases(lower, FILLER), n_words) * 100.0,
        f"{p}check_rate": safe_div(_count_phrases(lower, UNDERSTAND_CHECK), n_utt) * 100.0,
        f"{p}repetition_rate": _repetition_rate(words),
        f"{p}mean_words": safe_div(len(words), n_utt),
        f"{p}type_token_ratio": safe_div(len(set(words)), len(words)),
    }
    out.update(robust_stats(word_counts.to_numpy(dtype=float), f"{p}words"))
    return out


def _answer_length_trajectory(df: pd.DataFrame) -> dict[str, float]:
    """Does the student's answer length trend up or down across the session?

    A rising trajectory plausibly signals growing confidence/elaboration; a falling one
    signals disengagement. Fit on utterance order, normalized to the session length so
    it is comparable across sessions of different lengths.
    """
    stu = df[df["role"] == "student"]
    if len(stu) < 5:
        return {
            f"{PREFIX}student_len_slope": 0.0,
            f"{PREFIX}student_len_r2": 0.0,
            f"{PREFIX}student_len_last_first_ratio": 0.0,
        }
    y = stu["content"].str.len().to_numpy(dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    # least squares slope
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0
    pred = ym + slope * (x - xm)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    q = max(1, len(y) // 4)
    first, last = y[:q].mean(), y[-q:].mean()
    return {
        f"{PREFIX}student_len_slope": slope,
        f"{PREFIX}student_len_r2": float(r2),
        f"{PREFIX}student_len_last_first_ratio": safe_div(float(last), float(first)),
    }


def session_linguistic_features(sid: str, df: pd.DataFrame) -> dict[str, Any]:
    feats: dict[str, Any] = {"session_id": sid}
    for role in ("student", "tutor", "background"):
        feats.update(_role_linguistics(df.loc[df["role"] == role, "content"], role))
    feats.update(_answer_length_trajectory(df))

    # Session-level ASR quality: unclear density over ALL content.
    all_text = "\n".join(df["content"].tolist())
    n_utt = max(len(df), 1)
    feats[f"{PREFIX}unclear_density_all"] = len(_UNCLEAR_RE.findall(all_text)) / n_utt

    # Ratio features: tutor questioning vs student questioning is a dialogue-role signal.
    feats[f"{PREFIX}q_ratio_student_tutor"] = safe_div(
        feats[f"{PREFIX}student_q_rate"], feats[f"{PREFIX}tutor_q_rate"], default=0.0
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
    "features.linguistic",
    requires="cpu",
    max_tier="cpu",
    description="question/hedge/affirm markers + ASR disfluency, per role",
)
def build(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    stage_local()
    path = block_cache_path("linguistic", VERSION, subsample, source_hash=_source())

    def compute() -> pd.DataFrame:
        rows = []
        it = iter_session_frames(subsample=subsample)
        for sid, df in pbar(it, desc="features.linguistic", unit="session"):
            rows.append(session_linguistic_features(sid, df))
        return pd.DataFrame(rows)

    out = load_or_compute(path, compute, force=force, label="features.linguistic")
    return {
        "output_path": str(path),
        "n_sessions": int(len(out)),
        "n_features": int(out.shape[1] - 1),
    }
