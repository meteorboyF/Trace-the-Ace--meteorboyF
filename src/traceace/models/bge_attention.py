"""Supervised objective-conditioned attention over frozen BGE transcript windows.

The frozen semantic experiments that preceded this model reduced each response to cosine
statistics or unsupervised PCA coordinates. That asks a tree to discover mastery from a
representation that was never trained for the outcome. Here the learning objective acts
only as a query over the session's windows; objective identity is never fed directly to the
classifier, so the model cannot recreate a target-encoded difficulty lookup.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..cv import load_folds
from ..evaluate import experiment_dir, save_oof, score_frame
from ..io import LABEL_COL, load_train_features
from ..logging_utils import get_logger
from ..progress import pbar
from ..tasks import task

log = get_logger("model.bge_attention")


class WindowAttention:
    """Factory wrapper that keeps torch optional until this model is actually used."""

    @staticmethod
    def build(dim: int, hidden: int, dropout: float, base_rate: float):
        import torch

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.window_proj = torch.nn.Linear(dim, hidden, bias=False)
                self.query_proj = torch.nn.Linear(dim, hidden, bias=False)
                self.position_score = torch.nn.Sequential(
                    torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
                )
                output = torch.nn.Linear(hidden, 1)
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(2 * hidden + 5),
                    torch.nn.Linear(2 * hidden + 5, hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    output,
                )
                bias = math.log(base_rate / max(1.0 - base_rate, 1e-6))
                assert output.bias is not None
                torch.nn.init.constant_(output.bias, bias)

            def forward(self, windows, query, positions, mask):
                wp = torch.tanh(self.window_proj(windows))
                qp = torch.tanh(self.query_proj(query))
                logits = (wp * qp[:, None, :]).sum(-1) / math.sqrt(wp.shape[-1])
                logits = logits + self.position_score(positions[..., None]).squeeze(-1)
                logits = logits.masked_fill(~mask, -1e4)
                attention = torch.softmax(logits, dim=1)
                pooled = (attention[..., None] * wp).sum(1)

                cosine = (windows * query[:, None, :]).sum(-1).masked_fill(~mask, -1e4)
                count = mask.sum(1).clamp_min(1)
                cosine_mean = cosine.masked_fill(~mask, 0.0).sum(1) / count
                cosine_max = cosine.max(1).values
                top = torch.topk(cosine, k=min(3, cosine.shape[1]), dim=1).values
                top = torch.where(top > -1e3, top, torch.zeros_like(top)).mean(1)
                entropy = -(attention * torch.log(attention.clamp_min(1e-8))).sum(1)
                attended_pos = (attention * positions).sum(1)
                stats = torch.stack([cosine_mean, cosine_max, top, entropy, attended_pos], dim=1)
                joint = torch.cat([pooled, pooled * qp, stats], dim=1)
                return self.head(joint).squeeze(-1)

        return _Model()


@dataclass
class _Store:
    windows: np.ndarray
    positions: np.ndarray
    slices: dict[str, tuple[int, int]]
    objectives: dict[str, np.ndarray]
    dim: int


def _load_store(subsample: int | None) -> _Store:
    from ..config import get_config
    from ..features.window_embeddings import (
        DEFAULT_MODEL,
        lo_embedding_path,
        window_embedding_path,
    )

    cfg = get_config()
    model_name = str(cfg.get("embeddings", "alignment_model", default=DEFAULT_MODEL))
    win_path = window_embedding_path(model_name, subsample)
    lo_path = lo_embedding_path(model_name, subsample)
    if not win_path.is_file() or not lo_path.is_file():
        raise FileNotFoundError(
            "BGE caches are missing; run features.window_embeddings on an L4 and restore "
            "data/features + data/interim before training model.bge_attention"
        )
    win = pd.read_parquet(win_path).sort_values(["session_id", "window_idx"]).reset_index(drop=True)
    lo = pd.read_parquet(lo_path)
    cols = [c for c in win.columns if c.startswith("e")]
    # Arrow-backed parquet columns may expose read-only NumPy views. Own the memory so
    # torch.from_numpy is safe when batches are copied into their padded tensors.
    windows = np.array(win[cols], dtype=np.float32, order="C", copy=True)
    positions = np.array(win["centre_pos"], dtype=np.float32, copy=True)
    slices: dict[str, tuple[int, int]] = {}
    for sid, indices in win.groupby("session_id", sort=False).indices.items():
        idx = np.asarray(indices)
        slices[str(sid)] = (int(idx.min()), int(idx.max()) + 1)
    objectives = {
        str(row["learning_objective_id"]): np.array(row[cols], dtype=np.float32, copy=True)
        for _, row in lo.iterrows()
    }
    return _Store(windows, positions, slices, objectives, len(cols))


class _Dataset:
    def __init__(self, frame: pd.DataFrame, store: _Store):
        self.frame = frame.reset_index(drop=True)
        self.store = store

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        start, stop = self.store.slices[str(row["session_id"])]
        query = self.store.objectives[str(row["learning_objective_id"])]
        return (
            self.store.windows[start:stop],
            self.store.positions[start:stop],
            query,
            float(row[LABEL_COL]),
            str(row["response_id"]),
        )


def _collate(batch):
    import torch

    size = len(batch)
    max_windows = max(len(item[0]) for item in batch)
    dim = batch[0][0].shape[1]
    windows = torch.zeros((size, max_windows, dim), dtype=torch.float32)
    positions = torch.zeros((size, max_windows), dtype=torch.float32)
    mask = torch.zeros((size, max_windows), dtype=torch.bool)
    query = torch.zeros((size, dim), dtype=torch.float32)
    labels = torch.empty(size, dtype=torch.float32)
    ids: list[str] = []
    for i, (w, p, q, y, rid) in enumerate(batch):
        n = len(w)
        windows[i, :n] = torch.from_numpy(w)
        positions[i, :n] = torch.from_numpy(p)
        mask[i, :n] = True
        query[i] = torch.from_numpy(q)
        labels[i] = y
        ids.append(rid)
    return windows, query, positions, mask, labels, ids


def _predict(model, loader, device) -> tuple[list[str], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ids: list[str] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for windows, query, positions, mask, y, batch_ids in loader:
            logits = model(
                windows.to(device), query.to(device), positions.to(device), mask.to(device)
            )
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
            ids.extend(batch_ids)
    return ids, np.concatenate(labels), np.concatenate(predictions)


@task(
    "model.bge_attention",
    requires="cpu",
    max_tier="a100",
    description="supervised LO-conditioned attention over cached BGE transcript windows",
)
def train(
    force: bool = False,
    subsample: int | None = None,
    experiment: str = "model.bge_attention",
    hidden: int = 96,
    dropout: float = 0.15,
    batch_size: int = 128,
    epochs: int = 20,
    patience: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    cv_seed: int | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train one attention model per session-disjoint fold and persist honest OOF."""
    import torch
    from torch.utils.data import DataLoader

    from ..config import get_config
    from ..evaluate import logloss

    cfg = get_config()
    seed = int(cfg.seed if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    store = _load_store(subsample)
    folds = load_folds(subsample=subsample, cv_seed=cv_seed)
    metadata = load_train_features()[["response_id", "session_id", "learning_objective_id"]]
    frame = folds.merge(metadata, on=["response_id", "session_id"], how="left")
    frame = frame[
        frame["session_id"].astype(str).isin(store.slices)
        & frame["learning_objective_id"].astype(str).isin(store.objectives)
    ].reset_index(drop=True)
    if len(frame) != len(folds):
        raise RuntimeError(
            f"BGE caches cover {len(frame)}/{len(folds)} responses; refusing partial training"
        )

    oof = np.full(len(frame), np.nan, dtype=float)
    fold_ids = sorted(int(value) for value in frame["fold"].unique())
    model_dir = experiment_dir(experiment, subsample, cv_seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_losses: list[float] = []
    best_epochs: list[int] = []

    for fold in pbar(fold_ids, desc="model.bge_attention folds", unit="fold"):
        train_frame = frame[frame["fold"] != fold]
        valid_frame = frame[frame["fold"] == fold]
        generator = torch.Generator().manual_seed(seed + fold)
        train_loader: Any = DataLoader(
            _Dataset(train_frame, store),  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=_collate,
            num_workers=0,
        )
        valid_loader: Any = DataLoader(
            _Dataset(valid_frame, store),  # type: ignore[arg-type]
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=_collate,
            num_workers=0,
        )
        model = WindowAttention.build(
            store.dim, hidden, dropout, float(train_frame[LABEL_COL].mean())
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for windows, query, positions, mask, y, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    windows.to(device), query.to(device), positions.to(device), mask.to(device)
                )
                loss = criterion(logits, y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            _, vy, vp = _predict(model, valid_loader, device)
            validation_loss = logloss(vy, vp)
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"fold {fold} never produced a finite model")
        model.load_state_dict(best_state)
        valid_ids, _, valid_predictions = _predict(model, valid_loader, device)
        positions_by_id = frame.reset_index().set_index("response_id")["index"]
        indices = positions_by_id.loc[valid_ids].to_numpy(dtype=int)
        oof[indices] = valid_predictions
        torch.save(
            {
                "state_dict": {key: value.cpu() for key, value in best_state.items()},
                "dim": store.dim,
                "hidden": hidden,
                "dropout": dropout,
                "fold": fold,
            },
            model_dir / f"fold{fold}.pt",
        )
        best_losses.append(best_loss)
        best_epochs.append(best_epoch)
        log.info("bge_attention fold %d: logloss=%.5f epoch=%d", fold, best_loss, best_epoch)
        del model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(oof).all():
        raise RuntimeError("BGE attention left OOF rows unpredicted")
    scored = frame.copy()
    scored["pred"] = oof
    save_oof(
        experiment, scored[["response_id", "session_id", LABEL_COL, "pred"]], subsample, cv_seed
    )
    manifest = {
        "experiment": experiment,
        "model": "BAAI/bge-small-en-v1.5",
        "objective_identity_feature": False,
        "dim": store.dim,
        "hidden": hidden,
        "dropout": dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "cv_seed": cv_seed,
        "folds": fold_ids,
        "best_epochs": best_epochs,
        "best_logloss": best_losses,
    }
    (model_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    result = score_frame(scored, experiment, subsample=subsample, cv_seed=cv_seed)
    result.update(
        {
            "output_path": str(model_dir),
            "device": device,
            "best_epochs": best_epochs,
            "fold_logloss": best_losses,
            "n_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in WindowAttention.build(
                        store.dim, hidden, dropout, float(frame[LABEL_COL].mean())
                    ).parameters()
                )
            ),
        }
    )
    return result
