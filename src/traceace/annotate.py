"""Tutoring-move annotation — DEV-TIME ONLY, never in the inference path.

**Rules position (important).** A generative LLM is used here to label a stratified
sample of *training* utterances with tutoring-move types. The resulting annotations are
training data for a small encoder classifier (:mod:`traceace.models.move_classifier`),
which is what actually runs at inference. Consequences:

* The submission never invokes a generative model — inference stays encoder-cheap.
* Test samples remain independently processed; nothing here touches the test set.
* Training with different or absent test data yields identical parameters, as required.

This is recorded as an ADR in docs/DECISIONS.md ("no generative model in the inference
path — dev-time annotation instead").

**Pluggable backends.** ``backend="heuristic"`` is a zero-cost rule-based labeller that
works on CPU and is the default so the pipeline and tests never depend on a GPU or an
API. ``backend="vllm"`` uses a local open-weights model (Apache-2.0/MIT only — see
docs/EXTERNAL_ASSETS.md) on Colab. Annotations are cached to Parquet and synced to
Drive; this task is intended to **run once**.

The taxonomy follows standard dialogue-act categories used in tutoring research, chosen
so an education researcher recognizes them without ML background.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_config
from .logging_utils import get_logger
from .paths import interim_dir, transcripts_dir
from .progress import pbar
from .staging import stage_local
from .tasks import task

log = get_logger("annotate")

VERSION = "v1"

# --- taxonomy ----------------------------------------------------------------
# Tutor moves. Deliberately small, legible, and grounded in tutoring literature.
TUTOR_MOVES = [
    "questioning",  # eliciting: asks the student to produce something
    "explaining",  # telling: delivers content/procedure
    "scaffolding",  # hints, breaks the problem down, prompts the next step
    "affirming",  # positive feedback / confirmation
    "correcting",  # negative feedback / repair
    "checking",  # explicit understanding check ("does that make sense?")
    "managing",  # logistics, greetings, off-task
]
STUDENT_MOVES = [
    "answering",
    "asking",
    "expressing_confusion",
    "acknowledging",
    "off_task",
]


def annotations_path(backend: str = "heuristic") -> Path:
    return interim_dir() / f"move_annotations_{backend}_{VERSION}.parquet"


# --- heuristic backend -------------------------------------------------------
_Q = re.compile(r"\?")
_CHECK = re.compile(
    r"\b(does that make sense|make sense|do you understand|is that clear|are you with me|"
    r"do you follow|got it|any questions)\b",
    re.I,
)
_AFFIRM = re.compile(
    r"\b(well done|good job|exactly|perfect|brilliant|excellent|correct|that's right|"
    r"spot on|lovely|fantastic|great|nice one|super)\b",
    re.I,
)
_CORRECT = re.compile(
    r"\b(not quite|not right|incorrect|that's wrong|almost|nearly|try again|"
    r"remember that|actually it)\b",
    re.I,
)
_SCAFFOLD = re.compile(
    r"\b(what if|let's try|first|next step|start by|think about|can you see|"
    r"what do we do|hint|step by step|break (it|this) down)\b",
    re.I,
)
_MANAGE = re.compile(
    r"\b(hello|hi there|good morning|good afternoon|bye|see you|can you hear|"
    r"your microphone|let's begin|welcome back|thank you for)\b",
    re.I,
)
_CONFUSION = re.compile(
    r"\b(i don'?t know|i'?m not sure|confused|don'?t get it|no idea|stuck|"
    r"what do you mean|i don'?t understand|pardon)\b",
    re.I,
)
_ACK = re.compile(r"^\s*(ok(ay)?|yeah|yes|yep|mm-?hmm|right|sure|got it|alright)[.!]?\s*$", re.I)


def heuristic_label(role: str, text: str) -> str:
    """Rule-based move label. Cheap, deterministic, and a sane bootstrap/fallback."""
    t = (text or "").strip()
    if role == "tutor":
        if _MANAGE.search(t):
            return "managing"
        if _CHECK.search(t):
            return "checking"
        if _CORRECT.search(t):
            return "correcting"
        if _AFFIRM.search(t):
            return "affirming"
        if _SCAFFOLD.search(t):
            return "scaffolding"
        if _Q.search(t):
            return "questioning"
        return "explaining"
    if role == "student":
        if _CONFUSION.search(t):
            return "expressing_confusion"
        if _Q.search(t):
            return "asking"
        if _ACK.match(t):
            return "acknowledging"
        if len(t) < 3:
            return "off_task"
        return "answering"
    return "managing"


# --- LLM backend (dev-time only) ---------------------------------------------
ANNOTATION_PROMPT = """You are labelling utterances from a K-12 maths tutoring session.
Label the TUTOR utterance with exactly one move type from this list:
{tutor_moves}

Definitions:
- questioning: asks the student to produce an answer or idea
- explaining: delivers content, a procedure, or a worked step
- scaffolding: gives a hint or decomposes the problem into a next step
- affirming: confirms or praises a correct contribution
- correcting: signals an error and repairs it
- checking: explicitly checks understanding
- managing: greetings, logistics, or off-task talk

Reply with the label only, lowercase, no punctuation.

Utterance: {text}
Label:"""


def _llm_label_batch(texts: list[str], model_name: str, roles: list[str]) -> list[str]:
    """Label a batch with a local open-weights model via vLLM (Colab GPU only)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="auto", gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=6)
    prompts = [
        ANNOTATION_PROMPT.format(tutor_moves=", ".join(TUTOR_MOVES), text=t[:1200]) for t in texts
    ]
    outs = llm.generate(prompts, params)
    labels = []
    valid = set(TUTOR_MOVES) | set(STUDENT_MOVES)
    for o, role, text in zip(outs, roles, texts):
        raw = o.outputs[0].text.strip().lower().split()
        lab = raw[0].strip(".,:;") if raw else ""
        labels.append(lab if lab in valid else heuristic_label(role, text))
    return labels


@task(
    "annotate.moves",
    requires="cpu",
    max_tier="a100",
    description="label a stratified sample of utterances with tutoring-move types (DEV ONLY)",
)
def moves(
    force: bool = False,
    subsample: int | None = None,
    backend: str = "heuristic",
    n_sample: int = 40000,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    seed: int | None = None,
) -> dict[str, Any]:
    """Produce ``move_annotations_<backend>_v1.parquet``: (session_id, utterance_id, role,
    content, move).

    ``backend="heuristic"`` (default) is CPU and free. ``backend="vllm"`` needs a GPU and
    is budgeted at ~40 units; it is intended to run exactly once.
    """
    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    path = annotations_path(backend)
    if path.is_file() and not force:
        log.warning("annotate.moves: CACHE HIT %s — skipping (force=True to redo)", path)
        df = pd.read_parquet(path)
        return {
            "output_path": str(path),
            "n_annotations": int(len(df)),
            "cached": True,
            "backend": backend,
        }

    stage_local()
    rng = np.random.default_rng(seed)
    files = sorted(transcripts_dir().glob("*.csv"))
    if subsample is not None:
        files = files[:subsample]
    # Stratified across sessions: sample a few utterances from many sessions rather than
    # many utterances from few, so the annotation set spans tutors and topics.
    per_session = max(1, n_sample // max(len(files), 1))

    rows: list[dict[str, Any]] = []
    for fp in pbar(files, desc=f"annotate.moves[{backend}] sample", unit="session"):
        try:
            df = pd.read_csv(fp, dtype=str)
        except Exception:
            continue
        df["role"] = df["role"].astype("string").str.strip().str.lower()
        df = df[df["role"].isin(["tutor", "student"])]
        if df.empty:
            continue
        take = min(per_session, len(df))
        idx = rng.choice(len(df), size=take, replace=False)
        sel = df.iloc[np.sort(idx)]
        for _, r in sel.iterrows():
            rows.append(
                {
                    "session_id": fp.stem,
                    "utterance_id": r.get("utterance_id"),
                    "role": r["role"],
                    "content": str(r.get("content") or ""),
                }
            )

    ann = pd.DataFrame(rows)
    if ann.empty:
        raise RuntimeError("no utterances sampled for annotation")

    if backend == "heuristic":
        ann["move"] = [heuristic_label(r, t) for r, t in zip(ann["role"], ann["content"])]
    elif backend == "vllm":
        ann["move"] = _llm_label_batch(ann["content"].tolist(), model_name, ann["role"].tolist())
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'heuristic' or 'vllm'")

    path.parent.mkdir(parents=True, exist_ok=True)
    ann.to_parquet(path, index=False)

    dist = ann.groupby(["role", "move"]).size().unstack(fill_value=0)
    res = {
        "output_path": str(path),
        "n_annotations": int(len(ann)),
        "backend": backend,
        "n_sessions_sampled": int(ann["session_id"].nunique()),
        "move_distribution": json.loads(dist.to_json()),
        "cached": False,
    }
    log.info(
        "annotate.moves[%s]: %d annotations over %d sessions",
        backend,
        len(ann),
        ann["session_id"].nunique(),
    )
    return res
