"""Offline inference for the fine-tuned transcript encoder — shipped inside submission.zip.

Importable two ways, exactly like ``inference_lib``:

* in the repo, as ``traceace.packaging.encoder_lib`` (training reuses the architecture so
  train and serve cannot drift), and
* at the zip root, as ``encoder_lib`` from ``main.py`` — hence **no relative imports and no
  traceace imports** anywhere in this file.

Runs inside the no-network container. Everything is loaded from local asset directories:
the tokenizer and model *config* are vendored by ``submission.build`` via
``save_pretrained``, and the weights come from our own per-fold checkpoints — so
``from_pretrained`` never needs the Hub. ``main.py`` additionally sets ``HF_HUB_OFFLINE``
before any transformers import, making a network attempt an error rather than a hang.

Asset layout under ``assets/encoder/``::

    encoder.json          # model_name, max_tokens, topk_windows, blend weight, n_folds
    tokenizer/            # AutoTokenizer.save_pretrained
    config/               # AutoConfig.save_pretrained (architecture only, no weights)
    fold0.pt … fold4.pt   # {"state_dict": fp16 best-epoch weights, "valid_auc": float}

Per-sample independence: each prediction is a function of that sample's rendered text and
training-fitted weights. Batching is a compute optimisation only — no value crosses rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_encoder_spec(encoder_dir: Path) -> dict[str, Any]:
    """Read and validate ``encoder.json``. Fails loudly on anything missing or malformed."""
    spec_path = Path(encoder_dir) / "encoder.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"{spec_path} missing — encoder assets are incomplete")
    spec = json.loads(spec_path.read_text())
    required = {"model_name", "max_tokens", "topk_windows", "blend_weight", "n_folds"}
    missing = required - set(spec)
    if missing:
        raise KeyError(f"encoder.json missing fields {sorted(missing)}")
    weight = float(spec["blend_weight"])
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"encoder blend_weight {weight} outside [0, 1]")
    if int(spec["n_folds"]) < 1:
        raise ValueError("encoder n_folds must be >= 1")
    return spec


def build_inference_model(encoder_dir: Path):
    """Reconstruct the training architecture from the vendored config (no weights yet).

    Mirrors ``transcript_encoder.build_model`` exactly: AutoModel backbone, mask-weighted
    mean-pool, single linear head. The attribute names (``encoder``, ``head``) must match the
    training module so the fold state dicts load key-for-key with ``strict=True`` — a missing
    or renamed key is a packaging bug and must be an error, never a silent partial load.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(Path(encoder_dir) / "config")

    class _Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_config(config)
            self.dropout = torch.nn.Dropout(0.0)  # inference: no-op, kept for key parity
            self.head = torch.nn.Linear(int(config.hidden_size), 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.head(self.dropout(pooled)).squeeze(-1)

    return _Encoder()


def _load_tokenizer(encoder_dir: Path):
    """Load the vendored tokenizer, surviving transformers-version skew.

    The 2026-08-22 submission failed on exactly this line: the build machine's newer
    transformers wrote a ``tokenizer_class`` name the container's older transformers does
    not know, and ``AutoTokenizer`` raised before anything loaded. The class name is
    metadata; the actual vocabulary and merges live in ``tokenizer.json``, which every
    transformers version can load directly through ``PreTrainedTokenizerFast``. So: try
    the polite route, and on ANY failure load the raw file — same tokens either way.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer_dir = Path(encoder_dir) / "tokenizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_dir / "tokenizer.json"))

    if tokenizer.pad_token is None:
        # Encoder vocabularies (ModernBERT-style) carry [PAD]; decoder ones use EOS.
        vocab = tokenizer.get_vocab()
        if "[PAD]" in vocab:
            tokenizer.pad_token = "[PAD]"
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise RuntimeError("vendored tokenizer has no pad token and no EOS to substitute")
    return tokenizer


def predict_probs(
    encoder_dir: Path,
    texts: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """Fold-averaged correctness probabilities for the rendered window texts.

    Folds are loaded one at a time — never five models in memory at once — and each fold
    scores every text before the next loads, so peak memory is one backbone regardless of
    fold count. Empty texts (unreadable transcripts) get probability NaN; the caller keeps
    its existing fallback path for those rows and blends only where a prediction exists.
    """
    import torch

    encoder_dir = Path(encoder_dir)
    spec = load_encoder_spec(encoder_dir)
    n_folds = int(spec["n_folds"])
    max_tokens = int(spec["max_tokens"])

    tokenizer = _load_tokenizer(encoder_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    scored = [i for i, text in enumerate(texts) if text.strip()]
    out = np.full(len(texts), np.nan, dtype=float)
    if not scored:
        return out

    fold_sums = np.zeros(len(scored), dtype=float)
    for fold in range(n_folds):
        ckpt_path = encoder_dir / f"fold{fold}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"{ckpt_path} missing — encoder assets are incomplete")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = build_inference_model(encoder_dir)
        # strict=True: a key mismatch means the packaged architecture differs from the
        # trained one, and a partial load would predict plausibly from random weights.
        model.load_state_dict(
            {k: v.to(torch.float32) for k, v in checkpoint["state_dict"].items()},
            strict=True,
        )
        model.to(device).eval()

        with torch.no_grad():
            for start in range(0, len(scored), batch_size):
                index_batch = scored[start : start + batch_size]
                encoded = tokenizer(
                    [texts[i] for i in index_batch],
                    truncation=True,
                    max_length=max_tokens,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
                ):
                    logits = model(encoded["input_ids"], encoded["attention_mask"])
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                fold_sums[start : start + len(index_batch)] += probs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out[np.asarray(scored, dtype=int)] = fold_sums / n_folds
    return out


def blend_with_base(
    base_probs: np.ndarray,
    encoder_probs: np.ndarray,
    weight: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Logit-space blend; rows where the encoder abstained (NaN) keep the base prediction."""
    base = np.clip(np.asarray(base_probs, dtype=float), eps, 1.0 - eps)
    enc = np.asarray(encoder_probs, dtype=float)
    have = np.isfinite(enc)
    if not have.any() or weight <= 0.0:
        return base
    enc_clipped = np.clip(np.where(have, enc, 0.5), eps, 1.0 - eps)
    base_logit = np.log(base / (1.0 - base))
    enc_logit = np.log(enc_clipped / (1.0 - enc_clipped))
    blended = 1.0 / (1.0 + np.exp(-((1.0 - weight) * base_logit + weight * enc_logit)))
    return np.where(have, blended, base)
