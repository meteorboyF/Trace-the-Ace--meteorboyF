"""Fine-tuned hierarchical transcript model for deployment-oriented experiments.

Unlike the frozen BGE experiments, gradients reach the language encoder. Long transcripts
are encoded as role-tagged chunks, pooled twice: once without the learning objective and once
with its text as a semantic query. The transcript-only auxiliary head and objective dropout
make objective-difficulty shortcuts costly rather than convenient.
"""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pandas as pd

from ..cv import load_folds
from ..evaluate import experiment_dir, save_oof, score_frame
from ..io import LABEL_COL, load_train_features, read_transcript
from ..logging_utils import get_logger
from ..progress import pbar
from ..robust_cv import load_robust_folds, purged_split_indices
from ..tasks import task

log = get_logger("model.hierarchical_transformer")

DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def render_transcript(frame: pd.DataFrame) -> str:
    roles = frame.get("role", pd.Series("unknown", index=frame.index)).fillna("unknown")
    content = frame.get("content", pd.Series("", index=frame.index)).fillna("")
    return "\n".join(f"<{role}> {text}" for role, text in zip(roles, content))


class HierarchicalEncoder:
    """Optional-torch factory used by training and lightweight unit tests."""

    @staticmethod
    def build(model_name: str, dropout: float = 0.15, gradient_checkpointing: bool = True):
        import torch
        from transformers import AutoModel

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                if gradient_checkpointing and hasattr(
                    self.encoder, "gradient_checkpointing_enable"
                ):
                    self.encoder.gradient_checkpointing_enable()
                hidden = int(self.encoder.config.hidden_size)
                self.transcript_score = torch.nn.Linear(hidden, 1)
                self.objective_score = torch.nn.Linear(hidden, hidden, bias=False)
                self.transcript_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden, 1),
                )
                self.conditional_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(3 * hidden),
                    torch.nn.Linear(3 * hidden, hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden, 1),
                )

            def _encode(self, ids, mask):
                return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

            def forward(
                self,
                chunk_ids,
                chunk_mask,
                valid_chunks,
                objective_ids,
                objective_mask,
                objective_dropout: float = 0.0,
            ):
                batch, chunks, length = chunk_ids.shape
                chunk_vec = self._encode(
                    chunk_ids.reshape(batch * chunks, length),
                    chunk_mask.reshape(batch * chunks, length),
                ).reshape(batch, chunks, -1)
                objective = self._encode(objective_ids, objective_mask)
                plain_logits = self.transcript_score(chunk_vec).squeeze(-1)
                plain_logits = plain_logits.masked_fill(~valid_chunks, -1e4)
                plain_weights = torch.softmax(plain_logits, dim=1)
                plain = (plain_weights[..., None] * chunk_vec).sum(1)
                transcript_logit = self.transcript_head(plain).squeeze(-1)

                if self.training and objective_dropout > 0:
                    keep = torch.rand(batch, 1, device=objective.device) >= objective_dropout
                    objective = objective * keep
                query = self.objective_score(objective)
                scores = (chunk_vec * query[:, None]).sum(-1) / max(query.shape[-1] ** 0.5, 1.0)
                scores = scores.masked_fill(~valid_chunks, -1e4)
                weights = torch.softmax(scores, dim=1)
                conditioned = (weights[..., None] * chunk_vec).sum(1)
                joint = torch.cat([conditioned, objective, conditioned * objective], dim=1)
                conditional_logit = self.conditional_head(joint).squeeze(-1)
                return transcript_logit, conditional_logit

        return _Model()


class _TranscriptDataset:
    def __init__(self, frame: pd.DataFrame, texts: dict[str, str]):
        self.frame = frame.reset_index(drop=True)
        self.texts = texts

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        objective = row.get("learning_objective")
        return {
            "response_id": str(row["response_id"]),
            "text": self.texts[str(row["session_id"])],
            "objective": "" if pd.isna(objective) else str(objective),
            "label": float(row[LABEL_COL]),
        }


class _Collator:
    def __init__(self, tokenizer, chunk_tokens: int, max_chunks: int, objective_tokens: int):
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.max_chunks = max_chunks
        self.objective_tokens = objective_tokens
        self.cache: dict[str, list[int]] = {}

    def _tokens(self, text: str) -> list[int]:
        if text not in self.cache:
            self.cache[text] = self.tokenizer.encode(text, add_special_tokens=False)
        return self.cache[text]

    def __call__(self, batch):
        import torch

        encoded: list[list[list[int]]] = []
        for item in batch:
            tokens = self._tokens(item["text"])
            payload = max(8, self.chunk_tokens - 2)
            chunks = [tokens[i : i + payload] for i in range(0, len(tokens), payload)]
            chunks = chunks[-self.max_chunks :] or [[]]
            encoded.append(
                [
                    self.tokenizer.build_inputs_with_special_tokens(chunk)[: self.chunk_tokens]
                    for chunk in chunks
                ]
            )
        max_c = max(len(chunks) for chunks in encoded)
        pad = int(self.tokenizer.pad_token_id or 0)
        ids = torch.full((len(batch), max_c, self.chunk_tokens), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        valid = torch.zeros((len(batch), max_c), dtype=torch.bool)
        for i, chunks in enumerate(encoded):
            for j, chunk in enumerate(chunks):
                n = len(chunk)
                ids[i, j, :n] = torch.tensor(chunk)
                mask[i, j, :n] = 1
                valid[i, j] = True
        objectives = self.tokenizer(
            [item["objective"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.objective_tokens,
            return_tensors="pt",
        )
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        response_ids = [item["response_id"] for item in batch]
        return (
            ids,
            mask,
            valid,
            objectives["input_ids"],
            objectives["attention_mask"],
            labels,
            response_ids,
        )


def _load_texts(session_ids: list[str]) -> dict[str, str]:
    return {
        sid: render_transcript(read_transcript(sid))
        for sid in pbar(session_ids, desc="load transformer transcripts", unit="session")
    }


def _split_frames(
    frame: pd.DataFrame, split_mode: str, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_mode == "session":
        folds = load_folds()
        merged = frame.merge(folds[["response_id", "fold"]], on="response_id", how="inner")
        return merged[merged["fold"] != fold], merged[merged["fold"] == fold]
    folds = load_robust_folds(split_mode)
    merged = frame.merge(folds[["response_id", "fold"]], on="response_id", how="inner")
    if split_mode == "objective":
        robust = folds.set_index("response_id").loc[merged["response_id"]].reset_index()
        train_idx, valid_idx = purged_split_indices(robust, fold)
        return merged.iloc[train_idx], merged.iloc[valid_idx]
    if split_mode == "domain":
        return merged[merged["fold"] != fold], merged[merged["fold"] == fold]
    raise ValueError("split_mode must be session, objective, or domain")


def _predict(
    model, loader, device, amp_dtype, blend: float
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ids: list[str] = []
    ys, ps, plain_ps, conditional_ps = [], [], [], []
    with torch.no_grad():
        for chunks, masks, valid, obj, obj_mask, labels, response_ids in loader:
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                plain, conditional = model(
                    chunks.to(device),
                    masks.to(device),
                    valid.to(device),
                    obj.to(device),
                    obj_mask.to(device),
                )
                plain_prob = torch.sigmoid(plain)
                conditional_prob = torch.sigmoid(conditional)
                probs = (1 - blend) * plain_prob + blend * conditional_prob
            ys.append(labels.numpy())
            ps.append(probs.float().cpu().numpy())
            plain_ps.append(plain_prob.float().cpu().numpy())
            conditional_ps.append(conditional_prob.float().cpu().numpy())
            ids.extend(response_ids)
    return (
        ids,
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(plain_ps),
        np.concatenate(conditional_ps),
    )


@task(
    "model.hierarchical_transformer",
    requires="l4",
    max_tier="a100",
    description="fine-tune hierarchical ModernBERT with objective dropout and dual heads",
)
def train(
    force: bool = False,
    subsample: int | None = None,
    experiment: str = "model.hierarchical_transformer",
    model_name: str = DEFAULT_MODEL,
    split_mode: str = "objective",
    fold: int | None = 0,
    chunk_tokens: int = 512,
    max_chunks: int = 8,
    objective_tokens: int = 96,
    batch_size: int = 2,
    accumulation_steps: int = 16,
    epochs: int = 4,
    patience: int = 2,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    objective_dropout: float = 0.5,
    auxiliary_weight: float = 0.5,
    inference_blend: float = 0.5,
    seed: int = 1234,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("hierarchical transformer training requires a CUDA runtime")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    metadata = load_train_features()
    labels = load_folds()[["response_id", LABEL_COL]]
    frame = metadata.merge(labels, on="response_id", how="inner")
    if subsample is not None:
        sessions = frame["session_id"].drop_duplicates().head(subsample)
        frame = frame[frame["session_id"].isin(sessions)].copy()
    texts = _load_texts(frame["session_id"].astype(str).drop_duplicates().tolist())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = _Collator(tokenizer, chunk_tokens, max_chunks, objective_tokens)
    fold_ids = [fold] if fold is not None else list(range(5))
    all_oof: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    out_dir = experiment_dir(experiment)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in fold_ids:
        assert fold_id is not None
        train_frame, valid_frame = _split_frames(frame, split_mode, int(fold_id))
        train_loader: Any = DataLoader(
            _TranscriptDataset(train_frame, texts),  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )
        valid_loader: Any = DataLoader(
            _TranscriptDataset(valid_frame, texts),  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        model = HierarchicalEncoder.build(model_name).to(device)
        encoder_params: list[Any] = []
        head_params: list[Any] = []
        for name, parameter in model.named_parameters():
            (encoder_params if name.startswith("encoder.") else head_params).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": learning_rate},
                {"params": head_params, "lr": head_learning_rate},
            ],
            weight_decay=weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, epochs + 1):
            model.train()
            for step, (chunks, masks, valid, obj, obj_mask, y, _) in enumerate(train_loader, 1):
                with torch.autocast("cuda", dtype=amp_dtype):
                    plain, conditional = model(
                        chunks.to(device),
                        masks.to(device),
                        valid.to(device),
                        obj.to(device),
                        obj_mask.to(device),
                        objective_dropout,
                    )
                    loss = (
                        criterion(conditional, y.to(device))
                        + auxiliary_weight * criterion(plain, y.to(device))
                    ) / accumulation_steps
                loss.backward()
                if step % accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            ids, yv, pv, _, _ = _predict(model, valid_loader, device, amp_dtype, inference_blend)
            from ..evaluate import logloss

            val_loss = logloss(yv, pv)
            log.info(
                "hierarchical %s fold %d epoch %d: logloss=%.5f",
                split_mode,
                fold_id,
                epoch,
                val_loss,
            )
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {
                    k: v.detach().to("cpu", dtype=torch.float16).clone()
                    for k, v in model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"fold {fold_id} produced no finite checkpoint")
        model.load_state_dict(best_state)
        ids, yv, pv, plain_pv, conditional_pv = _predict(
            model, valid_loader, device, amp_dtype, inference_blend
        )
        all_oof.append(
            pd.DataFrame(
                {
                    "response_id": ids,
                    LABEL_COL: yv,
                    "pred": pv,
                    "pred_transcript": plain_pv,
                    "pred_conditional": conditional_pv,
                    "session_id": valid_frame.set_index("response_id")
                    .loc[ids, "session_id"]
                    .to_numpy(),
                }
            )
        )
        checkpoint = out_dir / f"{split_mode}_fold{fold_id}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "model_name": model_name,
                "fold": fold_id,
                "split_mode": split_mode,
                "chunk_tokens": chunk_tokens,
                "max_chunks": max_chunks,
                "objective_tokens": objective_tokens,
                "inference_blend": inference_blend,
            },
            checkpoint,
        )
        fold_metrics.append(
            {
                "fold": fold_id,
                "logloss": best_loss,
                "best_epoch": best_epoch,
                "n_train": len(train_frame),
                "n_valid": len(valid_frame),
            }
        )
        del model, optimizer, best_state
        torch.cuda.empty_cache()
    oof = pd.concat(all_oof, ignore_index=True)
    save_oof(f"{experiment}.{split_mode}", oof)
    if len(oof) == len(frame):
        result = score_frame(oof, f"{experiment}.{split_mode}")
    else:
        # A staged one-fold run is intentionally not a complete OOF cohort, so comparing it
        # with the full baseline would be an unpaired score and ``score_frame`` rightly
        # refuses. Report only metrics supported by this held-out cohort.
        from ..evaluate import auc, logloss

        y = oof[LABEL_COL].to_numpy(dtype=float)
        prediction = oof["pred"].to_numpy(dtype=float)
        result = {
            "experiment": f"{experiment}.{split_mode}",
            "logloss": logloss(y, prediction),
            "auc": auc(y, prediction),
            "n": len(oof),
            "pos_rate": float(y.mean()),
            "partial_oof": True,
        }
    from ..evaluate import auc, logloss

    y_all = oof[LABEL_COL].to_numpy(dtype=float)
    result["head_metrics"] = {
        "transcript": {
            "logloss": logloss(y_all, oof["pred_transcript"].to_numpy(dtype=float)),
            "auc": auc(y_all, oof["pred_transcript"].to_numpy(dtype=float)),
        },
        "conditional": {
            "logloss": logloss(y_all, oof["pred_conditional"].to_numpy(dtype=float)),
            "auc": auc(y_all, oof["pred_conditional"].to_numpy(dtype=float)),
        },
        "blend": {"logloss": result["logloss"], "auc": result["auc"]},
    }
    manifest = {
        "experiment": experiment,
        "model_name": model_name,
        "split_mode": split_mode,
        "folds": fold_ids,
        "objective_dropout": objective_dropout,
        "auxiliary_weight": auxiliary_weight,
        "inference_blend": inference_blend,
        "fold_metrics": fold_metrics,
        "result": result,
    }
    (out_dir / f"{split_mode}_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    result.update(
        {
            "output_path": str(out_dir),
            "fold_metrics": fold_metrics,
            "split_mode": split_mode,
            "model_name": model_name,
        }
    )
    return result
