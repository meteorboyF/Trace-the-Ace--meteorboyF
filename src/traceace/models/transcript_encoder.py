"""Fine-tuned encoder over the objective-relevant slice of a tutoring transcript.

**Why this exists.** 93% of our gap to the top of the leaderboard is discrimination, not
calibration (docs/ENDGAME.md §1). Hand-crafted lexical features over ASR dialogue have
plateaued at AUROC 0.606 on unseen objectives; the leaders sit at 0.63–0.64. That difference
is what a competent transformer fine-tune on dialogue looks like, and nothing in the feature
ladder is going to produce it.

**What is different from the rejected hierarchical pilot.** That model scored AUROC 0.4404 —
*below random* — which is a broken head, not a refuted hypothesis. It pooled chunk embeddings
twice and blended two heads, and it was judged against an objective-difficulty baseline that
does not exist on the leaderboard. This module is deliberately boring by comparison:

* **One example per response, not per session.** 58.8% of responses share a session with
  another response and 26% of outcome variance lives *between objectives inside one session*
  (docs/DATA.md). A session-level representation cannot address any of it.
* **Retrieval, not truncation.** Each example is the top-k windows most relevant to *this*
  objective, in transcript order, via the same ``topk_spans`` the feature blocks and the
  submission use — so training and inference select identical text by construction (ADR-007).
* **A flat encoder and a single head.** Mean-pool over tokens, one linear layer, one loss.

**The objective-text switch (``include_objective``) is the experiment, not a detail.**
Handing the model the objective description lets it memorise per-objective difficulty, which
is exactly the signal that scores 0.706 in session CV and 0.500 on the leaderboard — and it
is the organisers' stated anti-goal. Default is **off**: the encoder reads dialogue only, and
objective difficulty enters separately through ``features/lo_difficulty.py`` where it can be
measured. Set it True to quantify the shortcut rather than to ship it.

**Validation.** ``split_mode="objective"`` uses purged objective folds and reports
within-fold AUROC. Do not promote on the pooled figure: objective folds differ in base rate,
so pooling manufactures ranking signal (see :mod:`traceace.objective_eval`).

**Colab survival.** Folds are trained independently and each writes its predictions as soon
as it finishes. A disconnected runtime costs at most one fold: re-running the task skips
folds whose predictions are already on disk. The full OOF is assembled automatically once
every fold is present.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..cv import load_folds
from ..evaluate import experiment_dir, experiment_name, save_oof, score_frame
from ..io import LABEL_COL, load_train_features, read_transcript, write_parquet
from ..logging_utils import get_logger
from ..packaging.inference_lib import (
    frame_from_spans,
    normalize_frame,
    render_windows,
    topk_spans,
)
from ..progress import heartbeat, pbar
from ..robust_cv import load_robust_folds, purged_split_indices
from ..tasks import task

log = get_logger("model.transcript_encoder")

# ModernBERT-base: Apache-2.0, 8192-token context, and already the configured embedding
# backbone, so the weights are cached and the licence question is settled
# (docs/EXTERNAL_ASSETS.md). Its long context means retrieval width is a free parameter.
DEFAULT_MODEL = "answerdotai/ModernBERT-base"

# Windows are 20 utterances with 50% overlap (inference_lib.WINDOW/STRIDE), and adjacent
# picks get merged, so k windows is not k*20 utterances.
#
# MEASURED over 400 sessions rather than assumed, at the corpus ratio of 3.45 chars/token —
# fraction of examples that fit in 2,048 tokens without truncation:
#
#     k=4  median 1,335 tok · p95 2,085 · 94% fit
#     k=6  median 1,885 tok · p95 2,820 · 62% fit
#     k=8  median 2,417 tok · p95 3,604 · 25% fit
#
# So k=4 at 2,048. Truncation is not a graceful degradation here: it drops the *end* of the
# retrieved dialogue, and the end is where the tutor's closing check for understanding lives
# — plausibly the most predictive moment in the window. An earlier default of k=8 at 1,024
# tokens would have trained the model on openings only, discarding the very thing retrieval
# was for.
#
# Raise k and max_tokens together, never k alone (k=6 wants 3,072), and re-measure with
# `build_examples(...).text.str.len().describe()`.
DEFAULT_TOPK_WINDOWS = 4
DEFAULT_MAX_TOKENS = 2048


@dataclass
class EncoderConfig:
    """Everything that changes what the model sees or learns, in one hashable place."""

    model_name: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    topk_windows: int = DEFAULT_TOPK_WINDOWS
    include_objective: bool = False
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 2
    batch_size: int = 8
    accumulation_steps: int = 2
    warmup_frac: float = 0.1
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    # Trades ~30% speed for ~halved activation memory. Required on L4 (24 GB) at this
    # sequence length; leave False on the A100, where the full batch fits comfortably.
    gradient_checkpointing: bool = False
    num_workers: int = 2

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_examples(feats: pd.DataFrame, topk_windows: int, include_objective: bool) -> pd.DataFrame:
    """One text per response: the objective-relevant windows of its session transcript.

    Sessions are read once and reused across the objectives that share them, because the
    transcript read dominates the cost of this function.
    """
    from ..features.lo_alignment import _window_texts, _windows, fit_lo_vectorizer

    vectorizer = fit_lo_vectorizer()
    rows: list[dict[str, Any]] = []
    missing = 0

    for session_id, group in pbar(
        list(feats.groupby("session_id")), desc="encoder: build examples", unit="session"
    ):
        try:
            # normalize_frame re-sorts by (utterance_idx, t_seconds) — the SAME canonical
            # order every feature block and the submission path use. Skipping it here would
            # make window spans index differently-ordered rows, so the encoder would train
            # on different text than inference selects. That is the parity class that broke
            # submission #1 (ADR-013), and it must go through the shared implementation.
            transcript = normalize_frame(read_transcript(str(session_id)))
        except (FileNotFoundError, OSError, ValueError):
            missing += len(group)
            continue
        spans = _windows(transcript)
        window_matrix = vectorizer.transform(_window_texts(transcript, spans)) if spans else None

        for record in group.itertuples(index=False):
            lo_text = str(getattr(record, "learning_objective", ""))
            selected = topk_spans(lo_text, vectorizer, window_matrix, spans, topk_windows)
            text = render_windows(frame_from_spans(transcript, selected))
            rows.append(
                {
                    "response_id": str(record.response_id),
                    "session_id": str(session_id),
                    # The objective is stored either way so a run can be re-rendered without
                    # re-reading transcripts; whether it reaches the model is decided below.
                    "objective": lo_text if include_objective else "",
                    "text": text,
                }
            )

    if missing:
        log.warning("encoder: %d response(s) had no readable transcript and were dropped", missing)
    if not rows:
        raise RuntimeError("no examples built — is the transcript directory staged?")
    return pd.DataFrame(rows)


class _Dataset:
    """Tokenises lazily so a 35k-example run does not hold every encoding in RAM."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, cfg: EncoderConfig):
        self.texts = frame["text"].tolist()
        self.objectives = frame["objective"].tolist()
        self.labels = frame[LABEL_COL].to_numpy(dtype=np.float32)
        self.ids = frame["response_id"].tolist()
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, str, float, str]:
        return (
            self.objectives[index],
            self.texts[index],
            float(self.labels[index]),
            self.ids[index],
        )


def _collate(batch: list[tuple[str, str, float, str]], tokenizer: Any, cfg: EncoderConfig):
    import torch

    objectives = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    ids = [b[3] for b in batch]

    if cfg.include_objective:
        # Pair encoding puts the objective in segment A and the dialogue in segment B, so
        # truncation eats the dialogue tail rather than the objective.
        encoded = tokenizer(
            objectives,
            texts,
            truncation="only_second",
            max_length=cfg.max_tokens,
            padding=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_tokens,
            padding=True,
            return_tensors="pt",
        )
    return encoded, labels, ids


def build_model(cfg: EncoderConfig, base_rate: float):
    """Encoder + mean-pool + linear head, with the head biased to the base rate.

    Initialising the output bias at ``logit(base_rate)`` means the model starts calibrated and
    spends its first steps learning signal instead of learning the intercept. With an AUROC
    ceiling near 0.65 that matters: the intercept is most of the loss.
    """
    import torch
    from transformers import AutoModel

    class _Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(cfg.model_name, attn_implementation="sdpa")
            if cfg.gradient_checkpointing and hasattr(
                self.encoder, "gradient_checkpointing_enable"
            ):
                self.encoder.gradient_checkpointing_enable()
            hidden = int(self.encoder.config.hidden_size)
            self.dropout = torch.nn.Dropout(cfg.dropout)
            self.head = torch.nn.Linear(hidden, 1)
            torch.nn.init.zeros_(self.head.weight)
            prior = min(max(base_rate, 1e-4), 1 - 1e-4)
            self.head.bias.data.fill_(float(math.log(prior / (1 - prior))))

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.head(self.dropout(pooled)).squeeze(-1)

    return _Encoder()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _ensure_folds(split_mode: str, subsample: int | None, fold_seed: int | None) -> None:
    """Build any missing fold table before training starts.

    A fresh GPU runtime has no fold tables, and ``cv.build`` / ``cv.robust_build`` are
    CPU-guarded tasks the tier guard would refuse on an attached L4/A100 — so without this,
    the very first GPU cell dies on FileNotFoundError after the operator already paid to
    attach the accelerator. Building folds takes ~1s; the underlying builder functions are
    called directly (not through ``tasks.run``) precisely because the guard is about wasted
    hours, not wasted seconds.
    """
    if split_mode == "session":
        try:
            load_folds(subsample=subsample)
        except FileNotFoundError:
            from ..cv import build as build_session_folds

            log.info("encoder: session folds missing (subsample=%s); building", subsample)
            build_session_folds(subsample=subsample)
    else:
        try:
            load_robust_folds(split_mode, subsample=subsample, fold_seed=fold_seed)
        except FileNotFoundError:
            from ..robust_cv import build as build_robust_folds

            log.info("encoder: %s folds missing; building", split_mode)
            build_robust_folds(kind=split_mode, subsample=subsample, fold_seed=fold_seed)


def _fold_partition(
    examples: pd.DataFrame,
    split_mode: str,
    fold: int,
    subsample: int | None,
    fold_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/validation rows for one fold under the requested split regime."""
    if split_mode == "session":
        folds = load_folds(subsample=subsample)
        merged = examples.merge(folds[["response_id", "fold"]], on="response_id", how="inner")
        return merged[merged["fold"] != fold], merged[merged["fold"] == fold]

    folds = load_robust_folds(split_mode, subsample=subsample, fold_seed=fold_seed)
    merged = examples.merge(folds[["response_id", "fold"]], on="response_id", how="inner")
    if split_mode == "domain":
        return merged[merged["fold"] != fold], merged[merged["fold"] == fold]
    if split_mode != "objective":
        raise ValueError("split_mode must be 'session', 'objective', or 'domain'")

    # Objective folds purge validation sessions AND validation objectives from training, so
    # the partition has to come from the fold table itself rather than a fold != k mask.
    aligned = folds.set_index("response_id").loc[merged["response_id"]].reset_index()
    train_idx, valid_idx = purged_split_indices(aligned, fold)
    return merged.iloc[train_idx], merged.iloc[valid_idx]


def _predict(model, loader, device, amp_dtype) -> tuple[list[str], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ids: list[str] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for encoded, labels, batch_ids in loader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                logits = model(encoded["input_ids"], encoded["attention_mask"])
            ps.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(labels.numpy())
            ids.extend(batch_ids)
    return ids, np.concatenate(ys), np.concatenate(ps)


def _fold_pred_path(directory: Any, fold: int):
    return directory / f"fold{fold}_predictions.parquet"


def _fold_config_path(directory: Any, fold: int):
    return directory / f"fold{fold}_config.json"


def _fold_provenance(cfg: EncoderConfig, split_mode: str, fold_seed: int | None) -> dict[str, Any]:
    """Everything that makes two folds' predictions comparable."""
    return {"config": cfg.to_dict(), "split_mode": split_mode, "fold_seed": fold_seed}


def _check_fold_provenance(directory: Any, folds: list[int], expected: dict[str, Any]) -> None:
    """Refuse to assemble an OOF out of folds trained under different configurations.

    The failure this prevents: an aborted run leaves folds 0–2 trained at one setting, a
    later run finishes 3–4 at another, and the assembled "OOF" silently mixes two models.
    Every downstream number — the gate, the blend weights, the projection — would then
    describe a model that does not exist.
    """
    for fold in folds:
        path = _fold_config_path(directory, fold)
        if not path.is_file():
            raise RuntimeError(
                f"fold {fold} has predictions but no config sidecar ({path.name}). It predates "
                "provenance tracking or was copied by hand — retrain it with force=True."
            )
        found = json.loads(path.read_text())
        if found != expected:
            raise RuntimeError(
                f"fold {fold} was trained under a different configuration than this run.\n"
                f"  on disk: {found}\n  this run: {expected}\n"
                "Retrain with force=True (or move the old experiment aside) — assembling "
                "mixed-config folds would produce an OOF of a model that does not exist."
            )


def _train_one_fold(
    examples: pd.DataFrame,
    fold: int,
    cfg: EncoderConfig,
    split_mode: str,
    subsample: int | None,
    seed: int,
    directory: Any,
    fold_seed: int | None = None,
) -> dict[str, Any]:
    """Train one fold and persist its validation predictions. Returns fold metrics."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    _seed_everything(seed + fold)
    train_df, valid_df = _fold_partition(examples, split_mode, fold, subsample, fold_seed)
    if train_df.empty or valid_df.empty:
        raise RuntimeError(f"fold {fold} has an empty train or validation partition")
    if not set(train_df["session_id"]).isdisjoint(set(valid_df["session_id"])):
        raise RuntimeError(f"fold {fold} leaks sessions between train and validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        # Decoder backbones (Qwen et al.) ship without a pad token; padding batches would
        # crash. EOS-as-pad is safe here because the attention mask excludes pad positions
        # from the mean-pool.
        tokenizer.pad_token = tokenizer.eos_token
    base_rate = float(train_df[LABEL_COL].mean())
    model = build_model(cfg, base_rate).to(device)

    def collate(batch):
        return _collate(batch, tokenizer, cfg)

    # `_Dataset` is a plain class rather than a torch.utils.data.Dataset subclass so this
    # module imports without torch installed (the CPU dev environment, and every unit test
    # that only exercises example construction). DataLoader only needs __len__/__getitem__.
    # Tokenization happens in the collate, so worker processes keep the GPU fed; on CPU
    # (unit tests, local sanity runs) workers cost more than they save.
    loader_kw: dict[str, Any] = (
        {"num_workers": cfg.num_workers, "pin_memory": True, "persistent_workers": True}
        if device.type == "cuda" and cfg.num_workers > 0
        else {}
    )
    train_loader: Any = DataLoader(
        _Dataset(train_df, tokenizer, cfg),  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        **loader_kw,
    )
    valid_loader: Any = DataLoader(
        _Dataset(valid_df, tokenizer, cfg),  # type: ignore[arg-type]
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        **loader_kw,
    )

    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    encoder_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.learning_rate},
            {"params": head_params, "lr": cfg.head_learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // cfg.accumulation_steps)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_frac)

    def _linear_warmup(step: int) -> float:
        # Plain linear warmup->decay in torch, deliberately NOT transformers'
        # get_linear_schedule_with_warmup: this module must survive a Transformers major
        # bump on Colab, and a scheduler is not worth an import dependency.
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _linear_warmup)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_predictions: pd.DataFrame | None = None
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (encoded, labels, _) in enumerate(
            pbar(train_loader, desc=f"encoder fold {fold} epoch {epoch}", unit="batch"), start=1
        ):
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                logits = model(encoded["input_ids"], encoded["attention_mask"])
                loss = loss_fn(logits.float(), labels) / cfg.accumulation_steps
            loss.backward()
            running += float(loss.item()) * cfg.accumulation_steps
            if step % cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        ids, y_true, y_pred = _predict(model, valid_loader, device, amp_dtype)
        epoch_auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(1, len(train_loader)),
                "valid_auc": epoch_auc,
            }
        )
        log.info(
            "encoder fold %d epoch %d: train_loss=%.5f valid_auc=%.4f",
            fold,
            epoch,
            history[-1]["train_loss"],
            epoch_auc,
        )
        # Select on validation AUROC rather than loss: AUROC is what converts into a
        # leaderboard position, and the loss is dominated by the intercept.
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_predictions = pd.DataFrame(
                {"response_id": ids, LABEL_COL: y_true, "pred": y_pred, "fold": fold}
            )
            # Snapshot the best-epoch weights (CPU, fp16 for floats only — halving integer
            # buffers corrupts them). Without this the submission would have to RETRAIN:
            # predictions alone cannot be packaged.
            best_state = {
                key: value.detach()
                .to("cpu", dtype=torch.float16 if value.is_floating_point() else value.dtype)
                .clone()
                for key, value in model.state_dict().items()
            }

    if best_predictions is None or best_state is None:
        raise RuntimeError(f"fold {fold} produced no predictions")

    # Written as soon as the fold finishes so a Colab disconnect costs one fold, not a run.
    # The config sidecar lands first: predictions without provenance are treated as
    # untrusted by the assembly guard, so the write order fails safe.
    _fold_config_path(directory, fold).write_text(
        json.dumps(_fold_provenance(cfg, split_mode, fold_seed), sort_keys=True)
    )
    torch.save(
        {"state_dict": best_state, "valid_auc": best_auc},
        directory / f"fold{fold}_model.pt",
    )
    write_parquet(best_predictions, _fold_pred_path(directory, fold))
    del model
    torch.cuda.empty_cache()
    return {
        "fold": fold,
        "valid_auc": round(best_auc, 5),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "history": history,
    }


@task(
    "model.transcript_encoder",
    requires="l4",
    max_tier="a100",
    description="fine-tune ModernBERT on the objective-relevant transcript windows (response-level)",
)
def train(
    force: bool = False,
    subsample: int | None = None,
    experiment: str = "model.transcript_encoder",
    split_mode: str = "objective",
    folds: list[int] | int | None = None,
    seed: int | None = None,
    fold_seed: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Fine-tune the transcript encoder.

    Parameters
    ----------
    folds:
        Which folds to train. ``None`` trains all five; ``0`` or ``[0]`` trains one, which is
        the way to smoke a configuration before committing five folds of GPU time. Folds
        already on disk are skipped unless ``force=True``.
    overrides:
        Any :class:`EncoderConfig` field, e.g. ``max_tokens=2048``, ``include_objective=True``.
    """
    from ..config import get_config
    from ..objective_eval import projected_lb, within_fold_auc
    from ..staging import stage_local

    stage_local()
    cfg_global = get_config()
    seed = int(seed if seed is not None else cfg_global.seed)

    unknown = set(overrides) - set(EncoderConfig().to_dict())
    if unknown:
        raise ValueError(
            f"unknown encoder setting(s) {sorted(unknown)}; valid: "
            f"{sorted(EncoderConfig().to_dict())}"
        )
    cfg = EncoderConfig(**{**EncoderConfig().to_dict(), **overrides})

    feats = load_train_features()
    if subsample is not None:
        keep = feats["session_id"].drop_duplicates().head(max(1, subsample))
        feats = feats[feats["session_id"].isin(keep)]

    from ..io import load_train_labels

    labels = load_train_labels()
    feats = feats.merge(labels[["response_id", LABEL_COL]], on="response_id", how="inner")

    directory = experiment_dir(experiment, subsample)
    directory.mkdir(parents=True, exist_ok=True)

    if isinstance(folds, int):
        fold_ids = [folds]
    elif folds is None:
        fold_ids = list(range(int(cfg_global.cv["n_splits"])))
    else:
        fold_ids = [int(f) for f in folds]

    pending = [f for f in fold_ids if force or not _fold_pred_path(directory, f).is_file()]
    if not pending:
        log.info("CACHE HIT: folds %s already trained; pass force=True to retrain", fold_ids)

    fold_results: list[dict[str, Any]] = []
    if pending:
        _ensure_folds(split_mode, subsample, fold_seed)
        with heartbeat("encoder: rendering examples"):
            examples = build_examples(feats, cfg.topk_windows, cfg.include_objective)
        examples = examples.merge(feats[["response_id", LABEL_COL]], on="response_id", how="inner")
        log.info(
            "encoder: %d examples, %d chars median text, config=%s",
            len(examples),
            int(examples["text"].str.len().median()),
            cfg.to_dict(),
        )
        for fold in pending:
            with heartbeat(f"encoder fold {fold}"):
                fold_results.append(
                    _train_one_fold(
                        examples, fold, cfg, split_mode, subsample, seed, directory, fold_seed
                    )
                )
            # Checkpoint to Drive after EVERY fold: tarred artifacts PLUS direct copies of
            # this fold's files. The direct copies are redundancy against exactly the
            # 2026-08-22 failure — a 2.4GB rollup upload dropped at teardown — because a
            # handful of individually-large files is the case Drive FUSE handles well.
            #
            # A sync failure is FATAL, not a warning. On 2026-08-22 the Drive mount died
            # mid-run, every sync "succeeded" into a phantom local directory, and four
            # trained folds evaporated with the runtime. Stopping after the current fold
            # loses nothing: remount Drive (the notebook setup cell) and re-run — finished
            # folds are skipped by name. (Local runs have no drive_root; sync is a no-op
            # there and cannot raise.)
            from ..config import get_config as _get_config
            from ..maintenance import sync_artifacts

            try:
                with heartbeat(f"encoder fold {fold}: sync to Drive"):
                    sync_artifacts()
                drive_root = _get_config().drive_root
                if drive_root is not None:
                    mirror = drive_root / "models_mirror" / directory.name
                    mirror.mkdir(parents=True, exist_ok=True)
                    for name in (
                        f"fold{fold}_model.pt",
                        f"fold{fold}_predictions.parquet",
                        f"fold{fold}_config.json",
                        "training_manifest.json",
                    ):
                        src = directory / name
                        if src.is_file():
                            shutil.copyfile(src, mirror / name)
            except Exception as exc:
                raise RuntimeError(
                    f"Drive sync FAILED after fold {fold}: {exc}. Stopping here so finished "
                    "folds are not silently at risk — every completed fold before this one "
                    "is safe on Drive. Remount Drive (setup cell) and re-run this cell to "
                    "resume; completed folds are skipped by name."
                ) from exc

    # --- assemble the OOF once every fold is present --------------------------------
    available = sorted(
        int(p.stem.split("_")[0].removeprefix("fold"))
        for p in directory.glob("fold*_predictions.parquet")
    )
    expected = list(range(int(cfg_global.cv["n_splits"])))
    manifest = {
        "experiment": experiment,
        "subsample": subsample,
        "split_mode": split_mode,
        "seed": seed,
        "fold_seed": fold_seed,
        "folds_trained": available,
        "config": cfg.to_dict(),
    }
    (directory / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    result: dict[str, Any] = {
        "experiment": experiment_name(experiment, subsample),
        "split_mode": split_mode,
        "config": cfg.to_dict(),
        "folds_trained_now": [r["fold"] for r in fold_results],
        "fold_results": fold_results,
        "folds_on_disk": available,
    }

    if fold_results:
        aucs = [r["valid_auc"] for r in fold_results]
        result["mean_fold_auc_this_run"] = round(float(np.mean(aucs)), 5)
        result["projected_lb_this_run"] = round(projected_lb(float(np.mean(aucs))), 5)

    if available != expected:
        result["status"] = (
            f"partial — folds {sorted(set(expected) - set(available))} still missing; "
            "OOF not written yet"
        )
        log.info("encoder: %s", result["status"])
        return result

    _check_fold_provenance(directory, expected, _fold_provenance(cfg, split_mode, fold_seed))
    parts = [pd.read_parquet(_fold_pred_path(directory, f)) for f in expected]
    oof = pd.concat(parts, ignore_index=True)
    oof = oof.merge(feats[["response_id", "session_id"]], on="response_id", how="left")
    if oof["response_id"].duplicated().any():
        raise RuntimeError("assembled OOF has duplicate response_ids; fold partitions overlap")
    oof_path_local = save_oof(experiment, oof, subsample=subsample)
    from ..config import get_config as _get_config

    drive_root = _get_config().drive_root
    if drive_root is not None:
        mirror = drive_root / "oof_mirror"
        mirror.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(oof_path_local, mirror / oof_path_local.name)
        log.info("encoder: OOF mirrored directly to Drive")

    scored = score_frame(oof, experiment, subsample=subsample)
    result.update(scored)
    result["status"] = "complete"

    # The OOF's `fold` column is whatever regime trained it. Only objective folds admit the
    # headline metric — a session-fold AUC labeled "within_objective_fold_auc" would be the
    # exact metric confusion this project just spent a day purging.
    if split_mode == "objective":
        try:
            mean_auc, sd_auc, per_fold, _ = within_fold_auc(oof)
        except RuntimeError as exc:
            # Folds under the AUC minimum (small subsamples). Everything is trained and the
            # OOF is saved — crashing HERE would report a fully-successful run as a failure.
            result["within_objective_fold_auc"] = None
            result["within_objective_fold_auc_note"] = str(exc)
        else:
            result.update(
                {
                    "within_objective_fold_auc": round(mean_auc, 5),
                    "within_objective_fold_auc_sd": round(sd_auc, 5),
                    "per_fold_auc": {k: round(v, 5) for k, v in sorted(per_fold.items())},
                    "projected_lb_logloss": round(projected_lb(mean_auc), 5),
                }
            )
            log.info(
                "encoder %s: within-fold AUC %.4f ± %.4f -> projected LB %.4f",
                experiment,
                mean_auc,
                sd_auc,
                result["projected_lb_logloss"],
            )
    else:
        result["within_objective_fold_auc"] = None
        result["within_objective_fold_auc_note"] = (
            f"trained with split_mode={split_mode!r}; the objective metric requires "
            "split_mode='objective' (score via evaluate.by_objective_fold for a triage bound)"
        )
    return result
