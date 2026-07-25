"""Exploratory data analysis tasks.

Three tasks, all CPU:

* ``eda.overview``       — shapes, label balance, responses-per-session, missingness,
                           learning-objective family counts.
* ``eda.transcripts``    — the single most important measurement in the project: exact
                           per-session character and **token** distributions, role balance,
                           and session durations. Streams the per-session CSVs so it stays
                           memory-light.
* ``eda.inference_budget`` — projects inference cost of candidate architectures against
                           the 6-hour A100 cap, using the measured token distribution.
                           Prints a go/no-go table. Architecture follows from this arithmetic.

Each task writes a JSON summary under ``runs/eda/`` and (where useful) a
publication-quality figure under ``artifacts/figures/``. The measured numbers are
transcribed into ``docs/DATA.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_config
from .io import LABEL_COL, load_submission_format, load_train_features, load_train_labels
from .logging_utils import get_logger
from .paths import figures_dir, iter_files, runs_dir, transcripts_dir
from .progress import pbar
from .staging import stage_local
from .tasks import task
from .viz import save_fig, setup_mpl

log = get_logger("eda")


def _eda_dir() -> Path:
    d = runs_dir() / "eda"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump(name: str, payload: dict[str, Any]) -> Path:
    path = _eda_dir() / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _describe(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}
    qs = [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]
    out = {f"p{q}": float(np.percentile(x, q)) for q in qs}
    out["mean"] = float(x.mean())
    out["std"] = float(x.std())
    out["n"] = int(x.size)
    return out


# ---------------------------------------------------------------------------
# eda.overview
# ---------------------------------------------------------------------------
@task(
    "eda.overview",
    requires="cpu",
    max_tier="cpu",
    description="shapes, label balance, responses-per-session, missingness",
)
def overview(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    feats = load_train_features()
    labels = load_train_labels()
    sub_full = load_submission_format(smoke=False)
    sub_smoke = load_submission_format(smoke=True)

    merged = feats.merge(labels, on="response_id", how="left")
    n_resp = len(feats)
    n_sessions = feats["session_id"].nunique()
    resp_per_session = feats.groupby("session_id").size().to_numpy()

    label_rate = float(labels["correct"].mean())
    n_missing_label = int(merged["correct"].isna().sum())

    lo_col = "learning_objective"
    lo_id_col = "learning_objective_id" if "learning_objective_id" in feats.columns else None
    n_unique_lo = int(feats[lo_col].nunique()) if lo_col in feats.columns else None
    n_unique_lo_id = int(feats[lo_id_col].nunique()) if lo_id_col else None

    missingness = {c: float(feats[c].isna().mean()) for c in feats.columns}

    result: dict[str, Any] = {
        "n_train_responses": int(n_resp),
        "n_train_sessions": int(n_sessions),
        "responses_per_session": _describe(resp_per_session),
        "label_positive_rate": label_rate,
        "n_missing_label": n_missing_label,
        "n_test_responses_full": int(len(sub_full)),
        "n_test_responses_smoke": int(len(sub_smoke)),
        "n_unique_learning_objective_text": n_unique_lo,
        "n_unique_learning_objective_id": n_unique_lo_id,
        "feature_columns": list(feats.columns),
        "missingness": missingness,
    }
    result["output_path"] = str(_dump("overview", result))

    # figure: responses-per-session distribution
    try:
        setup_mpl()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.2))
        vals, counts = np.unique(resp_per_session, return_counts=True)
        ax.bar(vals, counts, color="#4C72B0")
        ax.set_xlabel("responses per session")
        ax.set_ylabel("number of sessions")
        ax.set_title("Learning objectives assessed per tutoring session")
        save_fig(fig, figures_dir() / "eda_responses_per_session")
    except Exception as exc:  # figures are non-critical
        log.warning("overview figure skipped: %s", exc)

    log.info(
        "overview: %d responses over %d sessions, pos-rate=%.4f", n_resp, n_sessions, label_rate
    )
    return result


# ---------------------------------------------------------------------------
# eda.transcripts
# ---------------------------------------------------------------------------
_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken

        enc_name = get_config().get("tokenizer", "tiktoken_encoding", default="cl100k_base")
        _TOKENIZER = tiktoken.get_encoding(enc_name)
    return _TOKENIZER


def _session_stats_dir() -> Path:
    return _eda_dir()


def session_stats_path() -> Path:
    return _eda_dir() / "session_stats.parquet"


@task(
    "eda.transcripts",
    requires="cpu",
    max_tier="cpu",
    description="exact per-session char/token distributions, role balance, durations",
)
def transcripts(
    force: bool = False, subsample: int | None = None, count_tokens: bool = True
) -> dict[str, Any]:
    """Stream per-session transcript CSVs and measure size/role/time distributions.

    ``count_tokens=True`` uses tiktoken (cl100k_base) as a license-clean token proxy.
    Writes per-session stats to ``runs/eda/session_stats.parquet`` for reuse by the
    inference-budget task and slice analyses.
    """
    stage_local()
    cache = session_stats_path()
    if cache.is_file() and not force and subsample is None:
        log.info("transcripts: cache hit %s (force=True to redo)", cache)
        stats = pd.read_parquet(cache)
        return _summarize_session_stats(stats, cached=True)

    tdir = transcripts_dir()
    files = list(iter_files(tdir, "*.csv"))
    if subsample is not None:
        files = files[:subsample]
    if not files:
        raise FileNotFoundError(f"no transcripts under {tdir}")

    enc = _tokenizer() if count_tokens else None
    rows: list[dict[str, Any]] = []
    roles_seen: dict[str, int] = {}

    for path in pbar(files, desc="eda.transcripts scan", unit="session"):
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:
            log.debug("skip unreadable %s (%s)", path.name, exc)
            continue
        content = df.get("content")
        role = df.get("role")
        if content is None:
            continue
        text = content.fillna("").astype(str)
        role_l = (
            role.fillna("").astype(str).str.strip().str.lower()
            if role is not None
            else pd.Series([""] * len(text))
        )
        for r in role_l.unique():
            roles_seen[r] = roles_seen.get(r, 0) + int((role_l == r).sum())

        n_chars = int(text.str.len().sum())
        joined = "\n".join(text.tolist())
        n_tokens = len(enc.encode(joined)) if enc is not None else int(n_chars / 4)

        # timing (relative H:MM:SS -> seconds)
        from .data import parse_elapsed_seconds

        t = (
            df["timestamp"].map(parse_elapsed_seconds).to_numpy()
            if "timestamp" in df.columns
            else np.array([np.nan])
        )
        duration = float(np.nanmax(t)) if np.isfinite(t).any() else float("nan")

        stu = role_l == "student"
        tut = role_l == "tutor"
        rows.append(
            {
                "session_id": path.stem,
                "n_utterances": int(len(text)),
                "n_chars": n_chars,
                "n_tokens": int(n_tokens),
                "n_student_utt": int(stu.sum()),
                "n_tutor_utt": int(tut.sum()),
                "student_chars": int(text[stu].str.len().sum()) if stu.any() else 0,
                "tutor_chars": int(text[tut].str.len().sum()) if tut.any() else 0,
                "duration_s": duration,
            }
        )

    stats = pd.DataFrame(rows)
    dest = cache if subsample is None else _eda_dir() / f"session_stats_sub{subsample}.parquet"
    stats.to_parquet(dest, index=False)

    summary = _summarize_session_stats(stats, cached=False)
    summary["roles_seen"] = dict(sorted(roles_seen.items(), key=lambda kv: -kv[1]))
    summary["output_path"] = str(dest)
    _dump("transcripts", summary)

    # figure: token distribution
    try:
        setup_mpl()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.2))
        toks = stats["n_tokens"].to_numpy()
        ax.hist(toks, bins=60, color="#55A868")
        ax.axvline(np.median(toks), color="#C44E52", ls="--", label=f"median={np.median(toks):.0f}")
        ax.set_xlabel("tokens per session (tiktoken cl100k)")
        ax.set_ylabel("number of sessions")
        ax.set_title("Transcript length distribution")
        ax.legend()
        save_fig(fig, figures_dir() / "eda_token_distribution")
    except Exception as exc:
        log.warning("transcript figure skipped: %s", exc)

    log.info(
        "transcripts: %d sessions, median tokens=%.0f",
        len(stats),
        float(stats["n_tokens"].median()),
    )
    return summary


def _summarize_session_stats(stats: pd.DataFrame, cached: bool) -> dict[str, Any]:
    total_student = int(stats["n_student_utt"].sum())
    total_tutor = int(stats["n_tutor_utt"].sum())
    total_utt = int(stats["n_utterances"].sum())
    return {
        "n_sessions": int(len(stats)),
        "tokens_per_session": _describe(stats["n_tokens"].to_numpy()),
        "chars_per_session": _describe(stats["n_chars"].to_numpy()),
        "utterances_per_session": _describe(stats["n_utterances"].to_numpy()),
        "duration_seconds": _describe(stats["duration_s"].to_numpy()),
        "total_tokens_all_sessions": int(stats["n_tokens"].sum()),
        "role_utterance_counts": {
            "student": total_student,
            "tutor": total_tutor,
            "other": total_utt - total_student - total_tutor,
        },
        "cached": cached,
    }


# ---------------------------------------------------------------------------
# eda.inference_budget
# ---------------------------------------------------------------------------
# Candidate architectures with representative sustained throughput (tokens/sec) on
# one A100 80GB. These are deliberately conservative; verify empirically before trusting.
_CANDIDATES = [
    # name, throughput tok/s, note
    ("ModernBERT/DeBERTa encoder (chunked, fp16)", 150_000, "encode once, mean-pool"),
    ("Small encoder (MiniLM class)", 300_000, "very cheap, weaker signal"),
    ("7B generative LLM (full-transcript prefill)", 10_000, "vLLM prefill, no margin"),
    ("14B generative LLM (full-transcript prefill)", 5_000, "likely OVER cap"),
]
_CAP_HOURS = 6.0
_SAFE_HOURS = 4.5  # submission.verify rejects projected runtime above this


@task(
    "eda.inference_budget",
    requires="cpu",
    max_tier="cpu",
    description="go/no-go table: candidate architectures vs the 6h A100 cap",
)
def inference_budget(
    force: bool = False, subsample: int | None = None, encode_per: str = "session"
) -> dict[str, Any]:
    """Project inference cost against the 6h cap using measured token counts.

    ``encode_per``: ``"session"`` (encode each distinct test session once, reuse across
    its responses) or ``"response"`` (re-encode per response — the pessimistic bound).
    Both are reported. Requires ``eda.transcripts`` to have run.
    """
    sp = session_stats_path()
    if not sp.is_file():
        raise FileNotFoundError("run eda.transcripts first (need runs/eda/session_stats.parquet)")
    stats = pd.read_parquet(sp)
    mean_tok = float(stats["n_tokens"].mean())
    p95_tok = float(stats["n_tokens"].quantile(0.95))

    # test sizing + responses-per-session ratio (measured on train, applied to test)
    feats = load_train_features()
    resp_per_session = len(feats) / feats["session_id"].nunique()
    n_test_resp = int(len(load_submission_format(smoke=False)))
    n_test_sessions_est = int(round(n_test_resp / resp_per_session))

    n_units = n_test_sessions_est if encode_per == "session" else n_test_resp

    def project(mean_or_p95: float) -> list[dict[str, Any]]:
        table = []
        for name, tput, note in _CANDIDATES:
            total_tokens = n_units * mean_or_p95
            hours = total_tokens / tput / 3600.0
            verdict = (
                "GO" if hours <= _SAFE_HOURS else ("TIGHT" if hours <= _CAP_HOURS else "NO-GO")
            )
            table.append(
                {
                    "arch": name,
                    "throughput_tok_s": tput,
                    "note": note,
                    "total_tokens": int(total_tokens),
                    "hours": round(hours, 3),
                    "verdict": verdict,
                }
            )
        return table

    result = {
        "encode_per": encode_per,
        "mean_tokens_per_session": round(mean_tok, 1),
        "p95_tokens_per_session": round(p95_tok, 1),
        "responses_per_session": round(resp_per_session, 3),
        "n_test_responses": n_test_resp,
        "n_test_sessions_est": n_test_sessions_est,
        "n_encode_units": n_units,
        "cap_hours": _CAP_HOURS,
        "safe_hours": _SAFE_HOURS,
        "projection_at_mean": project(mean_tok),
        "projection_at_p95": project(p95_tok),
    }
    result["output_path"] = str(_dump("inference_budget", result))

    # pretty table to stdout (operator-facing; this is NOT the submission path)
    print("\nINFERENCE BUDGET — projected wall time vs 6h A100 cap")
    print(
        f"  encode_per={encode_per} · units={n_units} "
        f"(test_resp={n_test_resp}, est_sessions={n_test_sessions_est})"
    )
    print(f"  mean tok/session={mean_tok:.0f}  p95={p95_tok:.0f}")
    print(f"  {'arch':<44}{'tok/s':>9}{'hours@mean':>12}{'verdict':>9}")
    for r in result["projection_at_mean"]:
        print(f"  {r['arch']:<44}{r['throughput_tok_s']:>9}{r['hours']:>12.2f}{r['verdict']:>9}")
    log.info("inference_budget: mean=%.0f p95=%.0f tok/session", mean_tok, p95_tok)
    return result


# ---------------------------------------------------------------------------
# eda.lo_conditioning
# ---------------------------------------------------------------------------
@task(
    "eda.lo_conditioning",
    requires="cpu",
    max_tier="cpu",
    description="within-session label variance — how much does LO-conditioning matter?",
)
def lo_conditioning(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    """Decompose label variance into between- and within-session components.

    This is the measurement that decides whether session-level features can possibly
    suffice. A session-level representation assigns identical values to every response
    in a session, so the **within-session** share of variance is precisely the part such
    a model cannot explain, no matter how good it is. See docs/FINDINGS.md F1.
    """
    df = load_train_features().merge(load_train_labels(), on="response_id", how="inner")
    y = df[LABEL_COL].to_numpy(dtype=float)

    sizes = df.groupby("session_id").size()
    multi_ids = sizes[sizes > 1].index
    multi = df[df["session_id"].isin(multi_ids)]

    # variance decomposition, weighted by responses (single-response sessions
    # contribute zero within-session variance by construction)
    within_by_session = df.groupby("session_id")[LABEL_COL].var(ddof=0).fillna(0.0)
    counts = df.groupby("session_id").size()
    within_weighted = float((within_by_session * counts).sum() / counts.sum())
    total_var = float(np.var(y))

    nuniq = multi.groupby("session_id")[LABEL_COL].nunique()

    # Oracle floor for a session-only model: the best it can do is predict each
    # session's own mean. Nothing session-level can score below this.
    eps = 1e-15
    sess_mean = df.groupby("session_id")[LABEL_COL].transform("mean").clip(eps, 1 - eps)
    oracle_session = float(-np.mean(y * np.log(sess_mean) + (1 - y) * np.log(1 - sess_mean)))
    p = float(y.mean())
    prior_ll = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))

    result: dict[str, Any] = {
        "n_responses": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "n_multi_response_sessions": int(len(multi_ids)),
        "frac_sessions_multi_response": float(len(multi_ids) / df["session_id"].nunique()),
        "frac_responses_in_multi_sessions": float(len(multi) / len(df)),
        "n_multi_sessions_mixed_labels": int((nuniq > 1).sum()),
        "frac_multi_sessions_mixed_labels": float((nuniq > 1).mean()) if len(nuniq) else 0.0,
        "within_session_variance": within_weighted,
        "total_variance": total_var,
        "within_over_total_ratio": float(within_weighted / total_var) if total_var else 0.0,
        "prior_logloss": prior_ll,
        "oracle_session_only_logloss": oracle_session,
        "responses_per_session_counts": sizes.value_counts().sort_index().to_dict(),
    }
    result["output_path"] = str(_dump("lo_conditioning", result))
    log.info(
        "lo_conditioning: within/total variance=%.3f, %.1f%% of responses in multi-response "
        "sessions, %.1f%% of those mixed-label",
        result["within_over_total_ratio"],
        100 * result["frac_responses_in_multi_sessions"],
        100 * result["frac_multi_sessions_mixed_labels"],
    )
    return result


# ---------------------------------------------------------------------------
# eda.roles
# ---------------------------------------------------------------------------
@task(
    "eda.roles",
    requires="cpu",
    max_tier="cpu",
    description="what is in the `background` role? third speaker vs diarization failure",
)
def roles(
    force: bool = False, subsample: int | None = None, n_sessions: int = 300
) -> dict[str, Any]:
    """Characterize every role value, especially ``background``.

    Finding (docs/FINDINGS.md F3): ``background`` is **not** a third speaker and not pure
    noise — it is a speaker-diarization failure bucket containing genuine, misattributed
    pedagogical speech alongside backchannels and ``[unclear]`` markers. Its *volume* is
    therefore a usable data-quality feature, and dropping it would discard real teaching.
    """
    import re

    stage_local()
    rng = np.random.default_rng(get_config().seed)
    files = sorted(transcripts_dir().glob("*.csv"))
    if subsample is not None:
        n_sessions = min(n_sessions, subsample)
    if len(files) > n_sessions:
        files = [files[i] for i in rng.choice(len(files), size=n_sessions, replace=False)]

    unclear_re = re.compile(r"\[unclear\]", re.IGNORECASE)
    frames = []
    for fp in pbar(files, desc="eda.roles sample", unit="session"):
        try:
            frames.append(pd.read_csv(fp, dtype=str))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df["role"] = df["role"].astype("string").str.strip().str.lower()
    df["content"] = df["content"].astype("string").fillna("")

    by_role: dict[str, Any] = {}
    for role, grp in df.groupby("role"):
        text = grp["content"]
        lens = text.str.len()
        by_role[str(role)] = {
            "n_utterances": int(len(grp)),
            "share_of_utterances": float(len(grp) / len(df)),
            "median_chars": float(lens.median()),
            "mean_chars": float(lens.mean()),
            "max_chars": int(lens.max()),
            "frac_long_utterances_over_200_chars": float((lens > 200).mean()),
            "unclear_rate": float(
                text.map(lambda s: len(unclear_re.findall(s))).sum() / max(len(grp), 1)
            ),
            "n_distinct_strings": int(text.nunique()),
        }

    result: dict[str, Any] = {
        "n_sessions_sampled": len(files),
        "n_utterances": int(len(df)),
        "by_role": by_role,
        "roles_present": sorted(by_role),
    }
    # The diagnostic: substantive long-form speech inside `background` means diarization
    # failure rather than ambient noise.
    bg = by_role.get("background")
    if bg:
        result["background_verdict"] = (
            "diarization_failure_contains_real_speech"
            if bg["frac_long_utterances_over_200_chars"] > 0.01
            else "likely_noise_only"
        )
    result["output_path"] = str(_dump("roles", result))
    log.info("eda.roles: roles=%s", result["roles_present"])
    return result
