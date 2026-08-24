"""Domain-adaptive pretraining (DAPT): continue ModernBERT's MLM on our transcript corpus.

**The bet.** Every lever measured so far moves CV AUROC by ≤0.01. The gap to the top of the
leaderboard is ~0.035. DAPT is the only remaining lever with step-change upside: ModernBERT
has never seen ASR'd tutoring dialogue — ``[unclear]`` tokens, disfluencies, role-tagged
turns, redaction markers — and adapting its language model to this distribution before
fine-tuning is the standard remedy (Gururangan et al., "Don't Stop Pretraining"). Expected
outcome is honestly bimodal: somewhere between nothing and +0.02 CV. That is why it runs
through the same fold-0 gate as every other candidate before earning a full run.

**Compliance.** Trains on TRAINING transcripts only — unsupervised learning on the training
set is unrestricted; it is *test*-set transduction the rules prohibit. Deterministic seed.
The corpus is rendered with the SAME role-tagged format the fine-tune consumes
(``inference_lib.render_windows``), so pretraining adapts to the exact input distribution
the encoder will read — adapting to a different rendering would waste part of the budget.

**Durability.** Same posture as the encoder: the checkpoint saves and mirrors to Drive
after EVERY epoch, sync failure is fatal (a warning here converts an interruption into
silent total loss), and re-running resumes from the last completed epoch.

Output: a ``save_pretrained`` directory usable directly as ``model_name`` in
``model.transcript_encoder`` — fine-tuning from it needs no code changes anywhere else.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import numpy as np

from ..evaluate import experiment_dir
from ..io import load_train_features, read_transcript
from ..logging_utils import get_logger
from ..packaging.inference_lib import normalize_frame, render_windows
from ..progress import heartbeat, pbar
from ..tasks import task

log = get_logger("model.dapt")

DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def _render_corpus(subsample: int | None) -> list[str]:
    """One role-tagged text per session — full transcripts, not retrieval windows.

    MLM wants the whole distribution; retrieval selection would bias pretraining toward
    objective-adjacent dialogue and starve it of the openings/closings the fine-tune also
    reads at wider k.
    """
    feats = load_train_features()
    session_ids = feats["session_id"].drop_duplicates()
    if subsample is not None:
        session_ids = session_ids.head(max(1, subsample))

    texts: list[str] = []
    missing = 0
    for sid in pbar(session_ids.tolist(), desc="dapt: render corpus", unit="session"):
        try:
            frame = normalize_frame(read_transcript(str(sid)))
        except (FileNotFoundError, OSError, ValueError):
            missing += 1
            continue
        text = render_windows(frame)
        if text.strip():
            texts.append(text)
    if missing:
        log.warning("dapt: %d session(s) unreadable and skipped", missing)
    if not texts:
        raise RuntimeError("no transcripts rendered — is the data staged?")
    return texts


def _blockify(texts: list[str], tokenizer: Any, block_tokens: int, min_block: int) -> list[Any]:
    """Tokenize sessions and split into fixed-size blocks; drop tiny remainders.

    Each block is wrapped in the tokenizer's [CLS]/[SEP] (when defined) because that is the
    sequence format both the original pretraining AND our fine-tune use — adapting the model
    to unframed text would spend part of the budget on a format it never sees again.
    """
    import torch

    bos = tokenizer.cls_token_id
    eos = tokenizer.sep_token_id
    frame = int(bos is not None) + int(eos is not None)
    body = block_tokens - frame

    blocks: list[torch.Tensor] = []
    batch = 256
    for start in pbar(range(0, len(texts), batch), desc="dapt: tokenize corpus", unit="batch"):
        encoded = tokenizer(
            texts[start : start + batch], add_special_tokens=False, truncation=False
        )["input_ids"]
        for ids in encoded:
            for pos in range(0, len(ids), body):
                piece = ids[pos : pos + body]
                if len(piece) >= min_block:
                    framed = ([bos] if bos is not None else []) + list(piece)
                    if eos is not None:
                        framed.append(eos)
                    blocks.append(torch.tensor(framed, dtype=torch.long))
    if not blocks:
        raise RuntimeError("tokenization produced no blocks")
    return blocks


def _mirror_checkpoint(out_dir: Any) -> None:
    """Direct per-file Drive copy of the checkpoint. Fatal on failure, like encoder folds."""
    from ..config import get_config
    from ..maintenance import sync_artifacts

    sync_artifacts()
    drive_root = get_config().drive_root
    if drive_root is None:
        return
    mirror = drive_root / "models_mirror" / out_dir.name
    mirror.mkdir(parents=True, exist_ok=True)
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            shutil.copyfile(path, mirror / path.name)


@task(
    "model.dapt",
    requires="l4",
    max_tier="a100",
    description="domain-adaptive MLM pretraining of ModernBERT on the training transcripts",
)
def pretrain(
    force: bool = False,
    subsample: int | None = None,
    experiment: str = "dapt.base",
    model_name: str = DEFAULT_MODEL,
    block_tokens: int = 1024,
    min_block: int = 128,
    batch_size: int = 24,
    accumulation_steps: int = 2,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    warmup_frac: float = 0.06,
    mlm_probability: float = 0.30,
    max_grad_norm: float = 1.0,
    num_workers: int = 2,
    seed: int | None = None,
) -> dict[str, Any]:
    """Continue MLM pretraining; resumes from the last completed epoch on re-run.

    ``mlm_probability`` defaults to 0.30 — ModernBERT's own pretraining rate, so continued
    training matches the objective its weights were shaped by.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

    from ..config import get_config
    from ..staging import stage_local

    stage_local()
    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = experiment_dir(experiment, subsample)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "dapt_state.json"

    done_epochs = 0
    source = model_name
    if state_path.is_file() and not force:
        state = json.loads(state_path.read_text())
        done_epochs = int(state.get("epochs_done", 0))
        if state.get("config", {}) != {
            "model_name": model_name,
            "block_tokens": block_tokens,
            "mlm_probability": mlm_probability,
            "learning_rate": learning_rate,
            "seed": seed,
        }:
            raise RuntimeError(
                f"{experiment!r} has a checkpoint from a DIFFERENT configuration; pass "
                "force=True (discards it) or use a new experiment name"
            )
        if done_epochs >= epochs:
            log.info("CACHE HIT: %d epochs already trained at %s", done_epochs, out_dir)
            return {"experiment": experiment, "epochs_done": done_epochs, "output": str(out_dir)}
        source = str(out_dir)
        log.info("dapt: resuming from epoch %d at %s", done_epochs, out_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with heartbeat("dapt: corpus"):
        texts = _render_corpus(subsample)
        blocks = _blockify(texts, tokenizer, block_tokens, min_block)
    total_tokens = sum(len(b) for b in blocks)
    log.info(
        "dapt: %d sessions -> %d blocks of <=%d tokens (%.1fM tokens)",
        len(texts),
        len(blocks),
        block_tokens,
        total_tokens / 1e6,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(source, attn_implementation="sdpa").to(device)
    model.train()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    loader_kw: dict[str, Any] = (
        {"num_workers": num_workers, "pin_memory": True}
        if device.type == "cuda" and num_workers > 0
        else {}
    )
    loader: Any = DataLoader(
        blocks,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda feats: collator([{"input_ids": f} for f in feats]),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kw,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, len(loader) // accumulation_steps)
    total_steps = max(1, steps_per_epoch * (epochs - done_epochs))
    warmup_steps = int(total_steps * warmup_frac)

    def _linear(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.05, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _linear)

    losses: list[float] = []
    if state_path.is_file() and done_epochs:
        losses = list(json.loads(state_path.read_text()).get("mlm_losses", []))
    for epoch in range(done_epochs + 1, epochs + 1):
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(
            pbar(loader, desc=f"dapt epoch {epoch}", unit="batch"), start=1
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                loss = model(**batch).loss / accumulation_steps
            loss.backward()
            running += float(loss.item()) * accumulation_steps
            seen += 1
            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        epoch_loss = running / max(1, seen)
        losses.append(epoch_loss)
        log.info("dapt epoch %d: mlm_loss=%.4f", epoch, epoch_loss)

        # Save + mirror EVERY epoch: a 60-90 minute epoch is exactly the granularity at
        # which the 2026-08-22 class of loss stops being survivable this close to deadline.
        with heartbeat(f"dapt epoch {epoch}: save + Drive sync"):
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            state_path.write_text(
                json.dumps(
                    {
                        "epochs_done": epoch,
                        "mlm_losses": losses,
                        "config": {
                            "model_name": model_name,
                            "block_tokens": block_tokens,
                            "mlm_probability": mlm_probability,
                            "learning_rate": learning_rate,
                            "seed": seed,
                        },
                    },
                    indent=2,
                )
            )
            try:
                _mirror_checkpoint(out_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Drive sync FAILED after dapt epoch {epoch}: {exc}. Stopping so the "
                    "completed epochs are not silently at risk; remount Drive and re-run "
                    "to resume from this epoch."
                ) from exc

    return {
        "experiment": experiment,
        "epochs_done": epochs,
        "mlm_losses": losses,
        "n_blocks": len(blocks),
        "total_tokens_m": round(total_tokens / 1e6, 1),
        "output": str(out_dir),
        "finetune_with": f'model_name="{out_dir}"',
    }
