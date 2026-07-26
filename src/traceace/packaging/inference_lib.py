"""Self-contained feature extraction for inference.

**This file is copied verbatim into submission.zip** and imported by ``main.py``. It
therefore must not import anything from ``traceace`` and must depend only on numpy,
pandas, scikit-learn and the standard library — all present in the competition runtime.

It mirrors the training-time feature blocks (``features/structural.py``,
``linguistic.py``, ``temporal.py``, ``lo_alignment.py``). Keeping one implementation that
both sides import removes the classic train/serve skew bug: if this file changes, both
paths change together. ``tests/test_inference_parity.py`` asserts the two produce
identical values on the same input.

Nothing in this module prints.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# --- regexes / lexicons (kept identical to features/linguistic.py) -----------
TS_RE = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?\s*$")
WORD_RE = re.compile(r"[a-z']+")
BRACKET_RE = re.compile(r"\[[^\]]{0,40}\]")
UNCLEAR_RE = re.compile(r"\[unclear\]", re.IGNORECASE)
FALSE_START_RE = re.compile(r"\b[a-z]{1,12}-(?:\s|$)", re.IGNORECASE)
SELFCORR_RE = re.compile(
    r"\b(i mean|sorry|no wait|actually|rather|let me rephrase|scratch that)\b", re.IGNORECASE
)

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

ROLES = ("tutor", "student", "background")
MAX_PLAUSIBLE_GAP_S = 120.0
WINDOW = 20
STRIDE = 10
TOPK = 3


# --- shared primitives -------------------------------------------------------
def parse_elapsed_seconds(ts: Any) -> float:
    if not isinstance(ts, str):
        return float("nan")
    m = TS_RE.match(ts)
    if not m:
        return float("nan")
    h, mm, ss, frac = m.groups()
    total = float(int(h) * 3600 + int(mm) * 60 + int(ss))
    if frac:
        total += float("0." + frac)
    return total


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("session_id", "utterance_id", "role", "content", "timestamp"):
        if col not in df.columns:
            df[col] = pd.NA
    df["role"] = df["role"].astype("string").fillna("unknown").str.strip().str.lower()
    df["content"] = df["content"].astype("string").fillna("")
    df["utterance_idx"] = pd.to_numeric(df["utterance_id"], errors="coerce")
    df["t_seconds"] = df["timestamp"].map(parse_elapsed_seconds)
    return df.sort_values(["utterance_idx", "t_seconds"], kind="stable").reset_index(drop=True)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b else default


def robust_stats(x: np.ndarray, prefix: str, trim: float = 0.1) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_median": 0.0,
            f"{prefix}_trimmean": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_p90": 0.0,
        }
    lo, hi = np.percentile(x, [trim * 100, (1 - trim) * 100])
    trimmed = x[(x >= lo) & (x <= hi)]
    return {
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_trimmean": float(trimmed.mean()) if trimmed.size else float(np.median(x)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
    }


def count_phrases(text_lower: str, phrases: set[str]) -> int:
    return sum(text_lower.count(p) for p in phrases)


def repetition_rate(words: list[str]) -> float:
    if len(words) < 2:
        return 0.0
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return reps / (len(words) - 1)


# --- structural block --------------------------------------------------------
def structural_features(df: pd.DataFrame) -> dict[str, float]:
    P = "struct_"
    role = df["role"]
    content = df["content"]
    lens = content.str.len().to_numpy(dtype=float)
    n = len(df)
    feats: dict[str, float] = {f"{P}n_utterances": float(n)}

    for r in ROLES:
        m = (role == r).to_numpy()
        cnt = int(m.sum())
        chars = float(lens[m].sum()) if cnt else 0.0
        feats[f"{P}n_{r}"] = float(cnt)
        feats[f"{P}chars_{r}"] = chars
        feats[f"{P}frac_utt_{r}"] = safe_div(cnt, n)
        feats[f"{P}mean_len_{r}"] = safe_div(chars, cnt)
        feats.update(robust_stats(lens[m] if cnt else np.array([]), f"{P}len_{r}"))

    total_chars = float(lens.sum())
    stu_chars = feats[f"{P}chars_student"]
    tut_chars = feats[f"{P}chars_tutor"]
    feats[f"{P}student_talk_ratio"] = safe_div(stu_chars, stu_chars + tut_chars)
    feats[f"{P}student_turn_ratio"] = safe_div(
        feats[f"{P}n_student"], feats[f"{P}n_student"] + feats[f"{P}n_tutor"]
    )
    feats[f"{P}total_chars"] = total_chars

    # Plain numpy: pandas shift() yields NA (ambiguous truth value) and pandas 3's
    # pyarrow booleans lack a cumsum kernel. Identical on pandas 2.x and 3.x.
    role_arr = role.fillna("").astype(str).to_numpy()
    neq = np.ones(n, dtype=bool)
    if n > 1:
        neq[1:] = role_arr[1:] != role_arr[:-1]
    changes = int(neq[1:].sum()) if n > 1 else 0
    feats[f"{P}role_switches"] = float(changes)
    feats[f"{P}switch_rate"] = safe_div(changes, n)

    run_ids = np.cumsum(neq)
    run_len = pd.Series(run_ids).groupby(run_ids).size().to_numpy(dtype=float)
    feats.update(robust_stats(run_len, f"{P}runlen"))
    feats[f"{P}background_char_frac"] = safe_div(feats[f"{P}chars_background"], total_chars)
    return feats


# --- linguistic block --------------------------------------------------------
def _role_linguistics(texts: pd.Series, role: str) -> dict[str, float]:
    p = f"ling_{role}_"
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
    stripped = BRACKET_RE.sub(" ", lower)
    words = WORD_RE.findall(stripped)
    n_words = max(len(words), 1)
    n_q = int(texts.str.contains(r"\?", regex=True, na=False).sum())
    n_excl = int(texts.str.contains(r"!", regex=True, na=False).sum())
    word_counts = texts.map(lambda s: len(WORD_RE.findall(BRACKET_RE.sub(" ", s.lower()))))

    out = {
        f"{p}q_rate": safe_div(n_q, n_utt),
        f"{p}excl_rate": safe_div(n_excl, n_utt),
        f"{p}unclear_rate": safe_div(len(UNCLEAR_RE.findall(joined)), n_utt),
        f"{p}false_start_rate": safe_div(len(FALSE_START_RE.findall(stripped)), n_utt),
        f"{p}selfcorr_rate": safe_div(len(SELFCORR_RE.findall(lower)), n_utt),
        f"{p}hedge_rate": safe_div(count_phrases(lower, HEDGE), n_words) * 100.0,
        f"{p}affirm_rate": safe_div(count_phrases(lower, AFFIRM), n_words) * 100.0,
        f"{p}negate_rate": safe_div(count_phrases(lower, NEGATE), n_words) * 100.0,
        f"{p}filler_rate": safe_div(count_phrases(lower, FILLER), n_words) * 100.0,
        f"{p}check_rate": safe_div(count_phrases(lower, UNDERSTAND_CHECK), n_utt) * 100.0,
        f"{p}repetition_rate": repetition_rate(words),
        f"{p}mean_words": safe_div(len(words), n_utt),
        f"{p}type_token_ratio": safe_div(len(set(words)), len(words)),
    }
    out.update(robust_stats(word_counts.to_numpy(dtype=float), f"{p}words"))
    return out


def _answer_length_trajectory(df: pd.DataFrame) -> dict[str, float]:
    P = "ling_"
    stu = df[df["role"] == "student"]
    if len(stu) < 5:
        return {
            f"{P}student_len_slope": 0.0,
            f"{P}student_len_r2": 0.0,
            f"{P}student_len_last_first_ratio": 0.0,
        }
    y = stu["content"].str.len().to_numpy(dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0
    pred = ym + slope * (x - xm)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    q = max(1, len(y) // 4)
    return {
        f"{P}student_len_slope": slope,
        f"{P}student_len_r2": float(r2),
        f"{P}student_len_last_first_ratio": safe_div(float(y[-q:].mean()), float(y[:q].mean())),
    }


def linguistic_features(df: pd.DataFrame) -> dict[str, float]:
    P = "ling_"
    feats: dict[str, float] = {}
    for role in ("student", "tutor", "background"):
        feats.update(_role_linguistics(df.loc[df["role"] == role, "content"], role))
    feats.update(_answer_length_trajectory(df))
    all_text = "\n".join(df["content"].tolist())
    feats[f"{P}unclear_density_all"] = len(UNCLEAR_RE.findall(all_text)) / max(len(df), 1)
    feats[f"{P}q_ratio_student_tutor"] = safe_div(
        feats[f"{P}student_q_rate"], feats[f"{P}tutor_q_rate"], default=0.0
    )
    return feats


# --- temporal block ----------------------------------------------------------
def _slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        return 0.0
    x = np.linspace(0.0, 1.0, y.size)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    return float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0


def temporal_features(df: pd.DataFrame) -> dict[str, float]:
    P = "temp_"
    feats: dict[str, float] = {}
    t = df["t_seconds"].to_numpy(dtype=float)
    role = df["role"].to_numpy()
    finite = np.isfinite(t)
    duration = float(np.nanmax(t) - np.nanmin(t)) if finite.any() else 0.0
    feats[f"{P}duration_s"] = duration
    feats[f"{P}frac_timestamped"] = safe_div(int(finite.sum()), len(df))

    gaps = np.diff(t)
    gaps = gaps[np.isfinite(gaps)]
    gaps = gaps[(gaps >= 0) & (gaps <= MAX_PLAUSIBLE_GAP_S)]
    feats.update(robust_stats(gaps, f"{P}gap"))
    feats[f"{P}n_long_pauses"] = float(np.sum(gaps > 10.0))
    feats[f"{P}long_pause_rate"] = safe_div(float(np.sum(gaps > 10.0)), gaps.size)

    lat = []
    for i in range(len(df) - 1):
        if role[i] == "tutor" and role[i + 1] == "student":
            d = t[i + 1] - t[i]
            if np.isfinite(d) and 0 <= d <= MAX_PLAUSIBLE_GAP_S:
                lat.append(d)
    lat_arr = np.array(lat, dtype=float)
    feats.update(robust_stats(lat_arr, f"{P}student_latency"))
    feats[f"{P}n_student_responses"] = float(lat_arr.size)
    feats[f"{P}student_latency_slope"] = _slope(lat_arr)
    feats[f"{P}utt_per_min"] = safe_div(len(df), duration / 60.0)

    if finite.sum() > 10 and duration > 0:
        tt = t[finite]
        lo, hi = float(np.nanmin(tt)), float(np.nanmax(tt))
        edges = [lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi]
        thirds = [float(np.sum((tt >= edges[i]) & (tt < edges[i + 1]))) for i in range(3)]
        total = max(sum(thirds), 1.0)
        for i, v in enumerate(thirds):
            feats[f"{P}utt_frac_third{i + 1}"] = v / total
    else:
        for i in range(3):
            feats[f"{P}utt_frac_third{i + 1}"] = 0.0
    return feats


# --- LO-alignment block ------------------------------------------------------
def windows(df: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    n = len(df)
    if n == 0:
        return []
    if n <= window:
        return [(0, n)]
    out = [(s, min(s + window, n)) for s in range(0, n - window + 1, stride)]
    if out[-1][1] < n:
        out.append((max(0, n - window), n))
    return out


def window_texts(df: pd.DataFrame, spans: list[tuple[int, int]]) -> list[str]:
    content = df["content"].tolist()
    return [" ".join(content[s:e]) for s, e in spans]


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    if x.sum() <= 0 or x.size < 2:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def _window_dialogue_features(
    df: pd.DataFrame, spans: list[tuple[int, int]], idxs: np.ndarray
) -> dict[str, float]:
    P = "lo_"
    if len(idxs) == 0:
        return {
            f"{P}kw_student_talk_ratio": 0.0,
            f"{P}kw_student_q_rate": 0.0,
            f"{P}kw_tutor_q_rate": 0.0,
            f"{P}kw_hedge_rate": 0.0,
            f"{P}kw_mean_student_words": 0.0,
            f"{P}kw_n_utterances": 0.0,
        }
    rows = [df.iloc[spans[int(i)][0] : spans[int(i)][1]] for i in idxs]
    sub = pd.concat(rows) if len(rows) > 1 else rows[0]
    stu = sub[sub["role"] == "student"]
    tut = sub[sub["role"] == "tutor"]
    stu_chars = float(stu["content"].str.len().sum())
    tut_chars = float(tut["content"].str.len().sum())
    stu_text = " ".join(stu["content"].tolist()).lower()
    stu_words = WORD_RE.findall(BRACKET_RE.sub(" ", stu_text))
    return {
        f"{P}kw_student_talk_ratio": safe_div(stu_chars, stu_chars + tut_chars),
        f"{P}kw_student_q_rate": safe_div(
            int(stu["content"].str.contains(r"\?", regex=True, na=False).sum()), len(stu)
        ),
        f"{P}kw_tutor_q_rate": safe_div(
            int(tut["content"].str.contains(r"\?", regex=True, na=False).sum()), len(tut)
        ),
        f"{P}kw_hedge_rate": safe_div(sum(stu_text.count(h) for h in HEDGE), max(len(stu_words), 1))
        * 100.0,
        f"{P}kw_mean_student_words": safe_div(len(stu_words), len(stu)),
        f"{P}kw_n_utterances": float(len(sub)),
    }


def lo_alignment_features(
    df: pd.DataFrame,
    lo_text: str,
    vectorizer: Any,
    window_matrix: Any,
    spans: list[tuple[int, int]],
    topk: int = TOPK,
) -> dict[str, float]:
    from sklearn.metrics.pairwise import cosine_similarity

    P = "lo_"
    if not spans or window_matrix is None:
        return {}
    lo_vec = vectorizer.transform([lo_text or ""])
    sims = cosine_similarity(lo_vec, window_matrix).ravel()
    n_win = len(sims)
    order = np.argsort(-sims)
    top = order[: min(topk, n_win)]
    best = int(order[0])
    centres = np.array([(s + e) / 2.0 for s, e in spans], dtype=float)
    pos = centres / max(len(df), 1)

    feats = {
        f"{P}sim_max": float(sims[best]),
        f"{P}sim_topk_mean": float(sims[top].mean()),
        f"{P}sim_mean": float(sims.mean()),
        f"{P}sim_std": float(sims.std()),
        f"{P}sim_sum": float(sims.sum()),
        f"{P}sim_frac_above_half_max": float(
            np.mean(sims >= 0.5 * sims[best]) if sims[best] > 0 else 0.0
        ),
        f"{P}best_pos": float(pos[best]),
        f"{P}topk_pos_mean": float(pos[top].mean()),
        f"{P}topk_pos_spread": float(pos[top].max() - pos[top].min()) if len(top) > 1 else 0.0,
        f"{P}n_windows": float(n_win),
        f"{P}sim_gini": gini(sims),
    }
    feats.update(_window_dialogue_features(df, spans, top))
    return feats


def all_session_features(df: pd.DataFrame) -> dict[str, float]:
    """Every session-level block at once (structural + linguistic + temporal)."""
    feats: dict[str, float] = {}
    feats.update(structural_features(df))
    feats.update(linguistic_features(df))
    feats.update(temporal_features(df))
    return feats


# ---------------------------------------------------------------------------
# Feedback block — tutor corrective feedback around student attempts.
# ---------------------------------------------------------------------------
# Hypothesis: mastery signal lives in how the tutor RESPONDS to student attempts.
# A student who needs three attempts and two corrections before an affirmation has
# probably not mastered the topic; one who is affirmed immediately probably has.
#
# Deliberately lexical and curated: an education researcher can read these lists and
# audit exactly what the model keys on. The move classifier can supersede this later,
# but a transparent version is worth having regardless (and is free on CPU).

AFFIRM_MARKERS = [
    "exactly",
    "well done",
    "that's right",
    "thats right",
    "correct",
    "perfect",
    "brilliant",
    "excellent",
    "spot on",
    "good job",
    "great job",
    "nice work",
    "well spotted",
    "absolutely",
    "you got it",
    "that's it",
    "thats it",
    "lovely",
    "fantastic",
    "super",
    "yes good",
    "good boy",
    "good girl",
]
CORRECTION_MARKERS = [
    "not quite",
    "not right",
    "close",
    "almost",
    "nearly",
    "try again",
    "have another go",
    "not exactly",
    "that's wrong",
    "thats wrong",
    "incorrect",
    "let's try",
    "lets try",
    "remember",
    "careful",
    "actually",
    "hmm no",
    "no,",
    "not really",
    "think again",
    "look again",
    "check that",
    "not the",
]
REEXPLAIN_MARKERS = [
    "so what we do",
    "the way to",
    "let me show",
    "let me explain",
    "we need to",
    "the rule is",
    "remember that",
    "so first",
    "step by step",
    "what we're doing",
    "the reason",
    "because when",
    "so if we",
    "you have to",
]


def _marker_hits(text_lower: str, markers: list[str]) -> int:
    return sum(text_lower.count(m) for m in markers)


def _classify_tutor_turn(text: str) -> str:
    """Coarse label for one tutor turn: affirm | correct | reexplain | other.

    Order matters: correction is checked before affirmation because "close, but..."
    and "almost right" contain affirmation-adjacent words while being corrective.
    """
    t = (text or "").lower()
    n_corr = _marker_hits(t, CORRECTION_MARKERS)
    n_aff = _marker_hits(t, AFFIRM_MARKERS)
    if n_corr > n_aff:
        return "correct"
    if n_aff > 0:
        return "affirm"
    if _marker_hits(t, REEXPLAIN_MARKERS) > 0 or len(t) > 200:
        return "reexplain"
    return "other"


def feedback_features(df: pd.DataFrame, prefix: str = "fb_") -> dict[str, float]:
    """Corrective-feedback features over an (already scoped) stretch of dialogue.

    Called twice: once over the whole session (``fbs_``) and once over the
    LO-relevant windows (``fb_``), so the block is both a session descriptor and a
    topic-conditioned one.
    """
    P = prefix
    roles = df["role"].to_numpy()
    texts = df["content"].astype(str).to_numpy()
    n = len(df)

    labels = [_classify_tutor_turn(t) if r == "tutor" else None for r, t in zip(roles, texts)]
    n_aff = sum(1 for x in labels if x == "affirm")
    n_corr = sum(1 for x in labels if x == "correct")
    n_reex = sum(1 for x in labels if x == "reexplain")
    n_tutor = int((roles == "tutor").sum())

    feats: dict[str, float] = {
        f"{P}n_affirm": float(n_aff),
        f"{P}n_correct": float(n_corr),
        f"{P}n_reexplain": float(n_reex),
        f"{P}affirm_rate": safe_div(n_aff, n_tutor),
        f"{P}correct_rate": safe_div(n_corr, n_tutor),
        f"{P}reexplain_rate": safe_div(n_reex, n_tutor),
        # THE headline ratio: how much of the feedback is corrective vs affirming?
        f"{P}corrective_ratio": safe_div(n_corr, n_corr + n_aff, default=0.0),
    }

    # --- student attempts before an affirmation -----------------------------
    # Walk the dialogue; count consecutive student turns preceding each tutor
    # affirmation. A high value means the student needed several goes.
    attempts_before_affirm: list[float] = []
    run = 0
    for r, lab in zip(roles, labels):
        if r == "student":
            run += 1
        elif r == "tutor":
            if lab == "affirm" and run > 0:
                attempts_before_affirm.append(float(run))
            if lab in ("affirm", "correct", "reexplain"):
                run = 0
    aba = np.array(attempts_before_affirm, dtype=float)
    feats[f"{P}mean_attempts_before_affirm"] = float(aba.mean()) if aba.size else 0.0
    feats[f"{P}max_attempts_before_affirm"] = float(aba.max()) if aba.size else 0.0
    feats[f"{P}n_affirmed_episodes"] = float(aba.size)

    # --- did the tutor re-explain right after a student attempt? ------------
    reexplain_after_student = 0
    correct_after_student = 0
    affirm_after_student = 0
    for i in range(1, n):
        if roles[i] == "tutor" and roles[i - 1] == "student":
            if labels[i] == "reexplain":
                reexplain_after_student += 1
            elif labels[i] == "correct":
                correct_after_student += 1
            elif labels[i] == "affirm":
                affirm_after_student += 1
    responded = reexplain_after_student + correct_after_student + affirm_after_student
    feats[f"{P}reexplain_after_attempt_rate"] = safe_div(reexplain_after_student, responded)
    feats[f"{P}correct_after_attempt_rate"] = safe_div(correct_after_student, responded)
    feats[f"{P}affirm_after_attempt_rate"] = safe_div(affirm_after_student, responded)

    # --- repair: did a correction eventually lead to an affirmation? --------
    # Pedagogically the most interesting quantity here — successful repair is the
    # signature of learning happening inside the lesson.
    repairs, unrepaired = 0, 0
    pending = False
    for lab in labels:
        if lab == "correct":
            if pending:
                unrepaired += 1
            pending = True
        elif lab == "affirm" and pending:
            repairs += 1
            pending = False
    if pending:
        unrepaired += 1
    feats[f"{P}repair_rate"] = safe_div(repairs, repairs + unrepaired)
    feats[f"{P}n_repairs"] = float(repairs)
    feats[f"{P}n_unrepaired_corrections"] = float(unrepaired)

    # --- where does the LAST correction sit? --------------------------------
    # A correction near the end (unrepaired) is a much worse sign than one early on
    # that was subsequently resolved.
    last_corr = -1
    last_aff = -1
    for i, lab in enumerate(labels):
        if lab == "correct":
            last_corr = i
        elif lab == "affirm":
            last_aff = i
    feats[f"{P}last_correction_pos"] = (
        safe_div(last_corr, max(n - 1, 1)) if last_corr >= 0 else -1.0
    )
    feats[f"{P}last_affirm_pos"] = safe_div(last_aff, max(n - 1, 1)) if last_aff >= 0 else -1.0
    # +1 => ended on an affirmation (good), -1 => ended on a correction (bad)
    feats[f"{P}ends_on_affirm"] = float(np.sign(last_aff - last_corr)) if n else 0.0

    # --- longest consecutive corrective streak ------------------------------
    streak = best = 0
    for lab in labels:
        if lab == "correct":
            streak += 1
            best = max(best, streak)
        elif lab in ("affirm", "reexplain"):
            streak = 0
    feats[f"{P}max_correction_streak"] = float(best)
    return feats


# ---------------------------------------------------------------------------
# Trajectory block — ORDER-sensitive features over the LO-aligned window.
# ---------------------------------------------------------------------------
# Every other block is an aggregate, and aggregates are order-blind: a student who
# struggles early then masters the topic, and one whose fluency degrades into confusion,
# produce *identical* means, rates and ratios. Those two students have opposite outcomes.
#
# So these features are explicitly about SHAPE over time within the topic-relevant window:
# trends, first-vs-last thirds, where the final correction falls, and run lengths.


def _linreg_slope(y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of y over a normalized 0..1 index, plus R^2."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 0.0, 0.0
    x = np.linspace(0.0, 1.0, y.size)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0:
        return 0.0, 0.0
    slope = float(((x - xm) * (y - ym)).sum() / denom)
    pred = ym + slope * (x - xm)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, float(r2)


def _thirds(n: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    a, b = n // 3, (2 * n) // 3
    return [(0, max(a, 1)), (a, max(b, a + 1)), (b, n)]


def trajectory_features(df: pd.DataFrame, prefix: str = "traj_") -> dict[str, float]:
    """Order-sensitive features over an (already scoped) stretch of dialogue."""
    P = prefix
    n = len(df)
    roles = df["role"].to_numpy() if n else np.array([])
    texts = df["content"].astype(str).to_numpy() if n else np.array([])

    feats: dict[str, float] = {}
    labels = [_classify_tutor_turn(t) if r == "tutor" else None for r, t in zip(roles, texts)]

    # --- student utterance-length trend across the window -------------------
    stu_mask = roles == "student" if n else np.array([], dtype=bool)
    stu_len = np.array([len(t) for t in texts[stu_mask]], dtype=float) if n else np.array([])
    slope, r2 = _linreg_slope(stu_len)
    feats[f"{P}student_len_slope"] = slope
    feats[f"{P}student_len_r2"] = r2
    feats[f"{P}n_student_turns"] = float(stu_len.size)

    # --- first / middle / last third -----------------------------------------
    # Rates computed per third so "struggled early, recovered" is distinguishable from
    # "started fine, fell apart" — identical in any whole-window average.
    for i, (s, e) in enumerate(_thirds(n), start=1):
        seg_labels = labels[s:e]
        seg_roles = roles[s:e] if n else np.array([])
        n_tutor = int((seg_roles == "tutor").sum()) if n else 0
        n_corr = sum(1 for x in seg_labels if x == "correct")
        n_aff = sum(1 for x in seg_labels if x == "affirm")
        feats[f"{P}correct_rate_t{i}"] = safe_div(n_corr, n_tutor)
        feats[f"{P}affirm_rate_t{i}"] = safe_div(n_aff, n_tutor)
        seg_stu = [len(t) for t, r in zip(texts[s:e], seg_roles) if r == "student"] if n else []
        feats[f"{P}mean_student_len_t{i}"] = safe_div(sum(seg_stu), len(seg_stu))

    # deltas: last third minus first third — the direction of travel
    feats[f"{P}correct_rate_delta"] = feats.get(f"{P}correct_rate_t3", 0.0) - feats.get(
        f"{P}correct_rate_t1", 0.0
    )
    feats[f"{P}affirm_rate_delta"] = feats.get(f"{P}affirm_rate_t3", 0.0) - feats.get(
        f"{P}affirm_rate_t1", 0.0
    )
    feats[f"{P}student_len_delta"] = feats.get(f"{P}mean_student_len_t3", 0.0) - feats.get(
        f"{P}mean_student_len_t1", 0.0
    )

    # --- where does the LAST corrective turn fall, relative to window end? ---
    last_corr = -1
    last_aff = -1
    for i, lab in enumerate(labels):
        if lab == "correct":
            last_corr = i
        elif lab == "affirm":
            last_aff = i
    denom = max(n - 1, 1)
    feats[f"{P}last_correction_from_end"] = (
        float((n - 1 - last_corr) / denom) if last_corr >= 0 else 1.0
    )
    feats[f"{P}last_affirm_from_end"] = float((n - 1 - last_aff) / denom) if last_aff >= 0 else 1.0

    # --- how did the window END? ---------------------------------------------
    # The closing exchange is the tutor's last read on whether the student has it.
    feats[f"{P}ends_on_affirm"] = 1.0 if (last_aff >= 0 and last_aff > last_corr) else 0.0
    feats[f"{P}ends_on_correction"] = 1.0 if (last_corr >= 0 and last_corr > last_aff) else 0.0

    # last student->tutor exchange in the window
    final_exchange = 0.0
    for i in range(n - 1, 0, -1):
        if roles[i] == "tutor" and roles[i - 1] == "student":
            final_exchange = {"affirm": 1.0, "correct": -1.0}.get(labels[i] or "", 0.0)
            break
    feats[f"{P}final_exchange_valence"] = final_exchange

    # --- run lengths: longest streak of student attempts with no affirmation --
    streak = best = 0
    for r, lab in zip(roles, labels):
        if r == "student":
            streak += 1
        elif lab == "affirm":
            best = max(best, streak)
            streak = 0
    best = max(best, streak)
    feats[f"{P}max_unaffirmed_student_run"] = float(best)

    # longest consecutive corrective streak, and consecutive affirm streak
    cs = cbest = as_ = abest = 0
    for lab in labels:
        if lab == "correct":
            cs += 1
            cbest = max(cbest, cs)
            as_ = 0
        elif lab == "affirm":
            as_ += 1
            abest = max(abest, as_)
            cs = 0
    feats[f"{P}max_correction_run"] = float(cbest)
    feats[f"{P}max_affirm_run"] = float(abest)
    return feats


# ---------------------------------------------------------------------------
# LO-position block — how a fixed lesson budget is divided between objectives.
# ---------------------------------------------------------------------------
# Sessions run ~43 minutes and average 1.54 assessed objectives, so objectives
# COMPETE for a fixed time budget. An objective that got the last four minutes of a
# lesson shared with two others is in a very different position from one that had the
# whole hour. None of that is visible to any block that looks only at content.
#
# These features are also a directly reportable finding about time allocation in
# multi-objective tutoring, independent of the model.


def lo_position_features(
    lo_window_spans: list[tuple[int, int]],
    all_lo_spans: list[list[tuple[int, int]]],
    n_utterances: int,
    t_seconds: np.ndarray,
    prefix: str = "lopos_",
) -> dict[str, float]:
    """Positional/allocation features for one objective within its session.

    Parameters
    ----------
    lo_window_spans:
        Top-k window spans for THIS objective (already merged, chronological).
    all_lo_spans:
        Top-k spans for EVERY objective assessed in this session, including this one —
        needed to compute ordinal position and competition.
    n_utterances / t_seconds:
        Session length in utterances, and per-utterance elapsed seconds (may be NaN).
    """
    P = prefix
    n = max(n_utterances, 1)
    feats: dict[str, float] = {f"{P}n_competing_los": float(len(all_lo_spans))}

    if not lo_window_spans:
        feats.update(
            {
                f"{P}centre_pos": 0.5,
                f"{P}start_pos": 0.5,
                f"{P}end_pos": 0.5,
                f"{P}utt_share": 0.0,
                f"{P}minute_share": 0.0,
                f"{P}gap_to_session_end_utt": 0.0,
                f"{P}gap_to_session_end_s": 0.0,
                f"{P}ordinal": 0.0,
                f"{P}ordinal_frac": 0.0,
                f"{P}duration_s": 0.0,
                f"{P}overlap_with_others": 0.0,
            }
        )
        return feats

    start = min(s for s, _ in lo_window_spans)
    end = max(e for _, e in lo_window_spans)
    centre = (start + end) / 2.0

    feats[f"{P}start_pos"] = start / n
    feats[f"{P}end_pos"] = end / n
    feats[f"{P}centre_pos"] = centre / n

    # share of the lesson's utterances devoted to this objective
    covered = sum(e - s for s, e in lo_window_spans)
    feats[f"{P}utt_share"] = safe_div(covered, n)

    # --- ordinal position among the session's objectives --------------------
    centres = []
    for spans in all_lo_spans:
        if spans:
            c = (min(s for s, _ in spans) + max(e for _, e in spans)) / 2.0
            centres.append(c)
    centres_sorted = sorted(centres)
    ordinal = sum(1 for c in centres_sorted if c < centre)
    feats[f"{P}ordinal"] = float(ordinal)
    feats[f"{P}ordinal_frac"] = safe_div(ordinal, max(len(centres_sorted) - 1, 1))

    # --- how much does this objective's region overlap the others? ----------
    # High overlap => the objectives were taught interleaved rather than in blocks.
    others = [sp for sp in all_lo_spans if sp is not lo_window_spans]
    overlap = 0
    for sp in others:
        for s2, e2 in sp:
            for s1, e1 in lo_window_spans:
                overlap += max(0, min(e1, e2) - max(s1, s2))
    feats[f"{P}overlap_with_others"] = safe_div(overlap, max(covered, 1))

    # --- clock-time features -------------------------------------------------
    t = np.asarray(t_seconds, dtype=float)
    finite = np.isfinite(t)
    if finite.any():
        t_end = float(np.nanmax(t))
        t_start = float(np.nanmin(t))
        total_s = max(t_end - t_start, 1.0)
        si = min(max(int(start), 0), len(t) - 1)
        ei = min(max(int(end) - 1, 0), len(t) - 1)
        w_start = t[si] if np.isfinite(t[si]) else t_start
        w_end = t[ei] if np.isfinite(t[ei]) else t_end
        feats[f"{P}duration_s"] = float(max(w_end - w_start, 0.0))
        feats[f"{P}minute_share"] = safe_div(float(max(w_end - w_start, 0.0)), total_s)
        # An objective taught right before the bell has no time for consolidation.
        feats[f"{P}gap_to_session_end_s"] = float(max(t_end - w_end, 0.0))
    else:
        feats[f"{P}duration_s"] = 0.0
        feats[f"{P}minute_share"] = 0.0
        feats[f"{P}gap_to_session_end_s"] = 0.0

    feats[f"{P}gap_to_session_end_utt"] = safe_div(n - end, n)
    return feats


# ---------------------------------------------------------------------------
# Shared window selection — used identically by training and inference.
# ---------------------------------------------------------------------------
def topk_spans(
    lo_text: str,
    vectorizer: Any,
    window_matrix: Any,
    spans: list[tuple[int, int]],
    topk: int = TOPK,
) -> list[tuple[int, int]]:
    """Top-k LO-relevant window spans, merged and in chronological order.

    Order matters for the feedback and trajectory blocks, which walk the dialogue
    sequentially looking for attempt -> correction -> affirmation episodes.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not spans or window_matrix is None:
        return []
    sims = cosine_similarity(vectorizer.transform([lo_text or ""]), window_matrix).ravel()
    top = np.argsort(-sims)[: min(topk, len(sims))]
    chosen = sorted(spans[int(i)] for i in top)
    merged: list[tuple[int, int]] = []
    for s, e in chosen:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def frame_from_spans(df: pd.DataFrame, spans: list[tuple[int, int]]) -> pd.DataFrame:
    """Concatenate the given spans of ``df`` in order."""
    if not spans:
        return df.iloc[0:0]
    parts = [df.iloc[s:e] for s, e in spans]
    return pd.concat(parts) if len(parts) > 1 else parts[0]


def lo_prior_values(lo_ids: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Apply one booster's training-fold LO target-encoding map."""
    values = dict(spec.get("values", {}))
    fallback = float(spec["fallback"])
    return np.asarray([float(values.get(str(value), fallback)) for value in lo_ids], dtype=float)


# ---------------------------------------------------------------------------
# Calibration — serialized as PLAIN NUMBERS, never as a pickled estimator.
# ---------------------------------------------------------------------------
# A smoke test in the competition container revealed the local scikit-learn (1.9.0) did
# not match the runtime's (1.8.0), so unpickling fitted estimators raised
# InconsistentVersionWarning: "may lead to breaking code or invalid results". Warnings that
# say *invalid results* are the silent-degradation class this project keeps getting bitten
# by, so the calibrator is now reduced to the handful of numbers that define it and applied
# with arithmetic here. No sklearn object crosses the pickle boundary, and no future version
# drift in the image can affect it.
#
#   platt    -> sigmoid(coef * logit(p) + intercept)
#   isotonic -> piecewise-linear interpolation over the fitted thresholds


def apply_calibration(cal: dict[str, Any], p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply a plain-number calibrator produced by ``export_calibrator``."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    method = cal.get("method", "none")
    if method == "platt":
        z = np.log(p / (1.0 - p))
        return 1.0 / (1.0 + np.exp(-(float(cal["coef"]) * z + float(cal["intercept"]))))
    if method == "isotonic":
        x = np.asarray(cal["x_thresholds"], dtype=float)
        y = np.asarray(cal["y_thresholds"], dtype=float)
        if x.size == 0:
            return p
        return np.interp(p, x, y)
    return p
