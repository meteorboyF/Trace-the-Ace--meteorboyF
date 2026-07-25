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
    df["role"] = df["role"].astype("string").str.strip().str.lower()
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
