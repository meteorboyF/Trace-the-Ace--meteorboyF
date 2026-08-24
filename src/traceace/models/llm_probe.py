"""Zero-shot LLM probe: score one objective fold with an open-weights LLM via vLLM.

**Why this exists — the class-jump hypothesis.** The top of the leaderboard sits at test
AUROC ≈ 0.643 while our best fine-tuned ModernBERT-base arm reads ≈ 0.61 on held-out
objectives, and every input-side lever we have measured (wider retrieval, DAPT, objective
text) moves ≤ 0.01. The submission runtime ships **vLLM on an A100 80GB** — the organizers
expect winning solutions to run LLMs. The one lever we have not measured is model *class*:
a 7B instruction-tuned LLM reading the same transcript windows.

This probe answers two questions for ~2 GPU-hours before any fine-tuning money is spent:

1. **Solo skill** — AUROC of zero-shot yes/no logprob scoring on a held-out objective fold.
2. **Decorrelation** — saved as an OOF frame, so ``evaluate.arm_diversity`` can measure its
   logit correlation against the encoder arms. A 0.58-solo arm at ρ≈0.5 is worth more to the
   blend than a 0.61-solo arm at ρ≈0.95 (submission #2 measured that the hard way).

**Compliance.** The model is open-weights (default Qwen2.5-7B-Instruct, Apache-2.0, on the
license allowlist) and runs *locally* on the Colab GPU — no competition data touches an API.
Prompts are never printed or logged; reporting is aggregates only.

**Method.** Reuses the encoder's exact example rendering (``build_examples`` → retrieval-
selected windows), asks one yes/no question, and reads P(correct) from the first generated
token's logprobs. Temperature 0; deterministic.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np

from ..evaluate import auc, save_oof
from ..io import LABEL_COL, load_train
from ..logging_utils import get_logger
from ..progress import heartbeat
from ..robust_cv import load_robust_folds
from ..tasks import task

log = get_logger("model.llm_probe")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_PROMPT = (
    "You are reviewing a one-on-one tutoring session. Below are excerpts from its "
    "voice transcript (roles are tagged; [unclear] marks unrecognized audio).\n\n"
    "Learning objective being taught:\n{objective}\n\n"
    "Transcript excerpts:\n{text}\n\n"
    "Immediately after this session the student answered a quiz question testing this "
    "learning objective. Based only on the dialogue above, did the student answer it "
    "correctly? Reply with exactly one word: Yes or No."
)


def _first_token_ids(tokenizer: Any, variants: list[str]) -> set[int]:
    """First token id of every surface variant — logprob keys are single token ids."""
    ids: set[int] = set()
    for text in variants:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.add(int(encoded[0]))
    return ids


def _prob_yes(
    logprobs: dict[int, Any] | None, yes_ids: set[int], no_ids: set[int]
) -> tuple[float, bool]:
    """P(yes) from one position's top-k logprobs. Returns ``(prob, found_both)``.

    A missing side is floored 4 nats under the worst reported candidate rather than at
    -inf, so one absent token cannot saturate the probability to exactly 0/1.
    """
    if not logprobs:
        return 0.5, False
    values = {int(k): float(getattr(v, "logprob", v)) for k, v in logprobs.items()}
    floor = min(values.values()) - 4.0
    yes_lp = max((values[i] for i in yes_ids if i in values), default=floor)
    no_lp = max((values[i] for i in no_ids if i in values), default=floor)
    found_both = any(i in values for i in yes_ids) and any(i in values for i in no_ids)
    peak = max(yes_lp, no_lp)
    y, n = np.exp(yes_lp - peak), np.exp(no_lp - peak)
    return float(y / (y + n)), found_both


@task(
    "model.llm_probe",
    requires="a100",
    max_tier="a100",
    description="zero-shot LLM (vLLM) yes/no scoring of one held-out objective fold",
)
def probe(
    force: bool = False,
    subsample: int | None = None,
    experiment: str = "probe.llm_zeroshot",
    model_name: str = DEFAULT_MODEL,
    fold: int = 0,
    topk_windows: int = 10,
    include_objective: bool = True,
    max_model_len: int = 6144,
    gpu_memory_utilization: float = 0.90,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score the validation rows of one objective fold; save predictions as an OOF frame.

    The saved frame carries only this fold's rows — ``evaluate.arm_diversity`` inner-joins
    on ``response_id``, so correlations against full-OOF arms come out on exactly the shared
    fold, which is the honest comparison.
    """
    from vllm import LLM, SamplingParams

    from ..config import get_config
    from ..maintenance import sync_artifacts
    from ..staging import stage_local
    from .transcript_encoder import build_examples

    stage_local()
    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)

    train = load_train()
    folds = load_robust_folds("objective", subsample=subsample)
    merged = train.merge(folds[["response_id", "fold"]], on="response_id", how="inner")
    valid = merged[merged["fold"] == int(fold)].reset_index(drop=True)
    if valid.empty:
        raise RuntimeError(f"objective fold {fold} matched no training rows")
    log.info("llm_probe: fold %d -> %d validation rows", fold, len(valid))

    examples = build_examples(valid, topk_windows=topk_windows, include_objective=True)
    examples = examples.merge(
        valid[["response_id", LABEL_COL, "fold"]], on="response_id", how="inner"
    )

    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        seed=seed,
        max_logprobs=25,
    )
    tokenizer = llm.get_tokenizer()
    yes_ids = _first_token_ids(tokenizer, ["Yes", " Yes", "yes", " yes", "YES"])
    no_ids = _first_token_ids(tokenizer, ["No", " No", "no", " no", "NO"])
    if not yes_ids or not no_ids:
        raise RuntimeError("tokenizer produced no Yes/No token ids; cannot score")

    # Token-budget the transcript so prompt + scaffold always fits max_model_len. The
    # scaffold (instructions + objective + chat template) is budgeted at 512 tokens.
    body_budget = max_model_len - 512
    prompts: list[str] = []
    truncated = 0
    for record in examples.itertuples(index=False):
        text = str(record.text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > body_budget:
            text = tokenizer.decode(ids[:body_budget])
            truncated += 1
        objective = str(record.objective) if include_objective else "(not provided)"
        content = _PROMPT.format(objective=objective, text=text)
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:  # chat template without a thinking switch (e.g. Qwen2.5)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(rendered)
    if truncated:
        log.info("llm_probe: %d/%d prompts token-truncated to fit", truncated, len(prompts))

    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=seed)
    with heartbeat("llm_probe: vLLM generate"):
        outputs = llm.generate(prompts, params, use_tqdm=False)

    probs = np.full(len(outputs), 0.5, dtype=float)
    missing_both = 0
    for i, out in enumerate(outputs):
        completion = out.outputs[0]
        logprobs = completion.logprobs[0] if completion.logprobs else None
        probs[i], found = _prob_yes(logprobs, yes_ids, no_ids)
        if not found:
            missing_both += 1

    oof = examples[["response_id", LABEL_COL, "fold"]].copy()
    oof["pred"] = probs
    oof_path_local = save_oof(experiment, oof, subsample=subsample)

    fold_auc = auc(oof[LABEL_COL].to_numpy(), probs)
    decisive = float(np.mean((probs > 0.65) | (probs < 0.35)))
    result = {
        "experiment": experiment,
        "model_name": model_name,
        "fold": int(fold),
        "n": int(len(oof)),
        "fold_auc": round(fold_auc, 5),
        "mean_pred": round(float(probs.mean()), 4),
        "decisive_rate": round(decisive, 4),
        "fallback_rows": int(missing_both),
        "truncated_prompts": int(truncated),
        "next": "run evaluate.arm_diversity against the encoder arms before spending on LoRA",
    }
    log.info(
        "llm_probe: fold %d auc=%.5f mean_pred=%.4f fallback=%d",
        fold,
        fold_auc,
        float(probs.mean()),
        missing_both,
    )

    sync_artifacts()
    drive_root = cfg.drive_root
    if drive_root is not None:
        mirror = drive_root / "oof_mirror"
        mirror.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(oof_path_local, mirror / oof_path_local.name)
        log.info("llm_probe: OOF mirrored directly to Drive")
    return result
