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
    """Read and validate one encoder's ``encoder.json``. Fails loudly on anything malformed."""
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


def load_ensemble_spec(encoder_root: Path) -> dict[str, Any]:
    """Load an N-encoder ensemble, or a single encoder in the legacy layout.

    Returns ``{"base_weight": float, "encoders": [{"dir": Path, ...spec}, ...]}``.

    Two layouts are supported because the single-encoder format already shipped a scored
    submission and must keep working unchanged:

    * **ensemble** — ``assets/encoder/manifest.json`` plus one subdirectory per encoder.
      Each encoder carries its OWN ``topk_windows``/``max_tokens``, because ensemble members
      differ precisely in how much dialogue they read.
    * **legacy single** — ``assets/encoder/encoder.json`` with the tokenizer, config and
      folds directly beneath it.

    Weights are simplex over (base model + every encoder), matching what ``ensemble.blend``
    optimises, so the packaged blend is arithmetically the OOF blend we measured.
    """
    encoder_root = Path(encoder_root)
    manifest_path = encoder_root / "manifest.json"
    if not manifest_path.is_file():
        spec = load_encoder_spec(encoder_root)
        weight = float(spec["blend_weight"])
        return {
            "base_weight": 1.0 - weight,
            "encoders": [{**spec, "dir": encoder_root, "weight": weight}],
        }

    manifest = json.loads(manifest_path.read_text())
    if "encoders" not in manifest or not manifest["encoders"]:
        raise ValueError("encoder manifest lists no encoders")
    encoders = []
    for entry in manifest["encoders"]:
        name = str(entry["name"])
        directory = encoder_root / name
        spec = load_encoder_spec(directory)
        encoders.append({**spec, "dir": directory, "name": name, "weight": float(entry["weight"])})

    base_weight = float(manifest["base_weight"])
    total = base_weight + sum(e["weight"] for e in encoders)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"ensemble weights sum to {total:.4f}, expected 1.0")
    if base_weight < 0 or any(e["weight"] < 0 for e in encoders):
        raise ValueError("ensemble weights must be non-negative")
    return {"base_weight": base_weight, "encoders": encoders}


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
    batch_size: int = 64,
) -> np.ndarray:
    """Fold-averaged correctness probabilities for the rendered window texts.

    Folds are loaded one at a time — never five models in memory at once — and each fold
    scores every text before the next loads, so peak memory is one backbone regardless of
    fold count. Empty texts (unreadable transcripts) get probability NaN; the caller keeps
    its existing fallback path for those rows and blends only where a prediction exists.

    ``batch_size`` defaults to 64 rather than 16 because inference runs under ``no_grad``
    with no optimiser state, so the container's A100 is nowhere near saturated at 16 — and
    throughput is what decides whether an ENSEMBLE fits the 6-hour cap. One encoder at
    batch 16 measured 1.44 h projected; three at that rate would not fit. An OOM here halves
    the batch and retries rather than killing the run, because the alternative is losing a
    submission slot to a memory guess.
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

        def _run_batch(index_batch: list[int], model=model) -> np.ndarray:
            # model bound as a default arg: this closure is redefined per fold, and a late
            # binding would silently score every fold with the LAST fold's weights.
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
            return torch.sigmoid(logits.float()).cpu().numpy()

        with torch.no_grad():
            start = 0
            current = batch_size
            while start < len(scored):
                index_batch = scored[start : start + current]
                try:
                    probs = _run_batch(index_batch)
                except torch.cuda.OutOfMemoryError:
                    if current == 1:
                        raise
                    current = max(1, current // 2)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue  # retry the SAME slice at the smaller size
                fold_sums[start : start + len(index_batch)] += probs
                start += len(index_batch)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out[np.asarray(scored, dtype=int)] = fold_sums / n_folds
    return out


def blend_ensemble(
    base_probs: np.ndarray,
    encoder_probs: list[np.ndarray],
    encoder_weights: list[float],
    base_weight: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Simplex-weighted logit blend over the base model plus N encoders.

    Per row, the weights are renormalised over the arms that actually produced a prediction.
    An encoder abstains (NaN) when the transcript was unreadable, and without renormalising
    that row would be blended toward logit 0 — a confident shove to p=0.5 on exactly the
    rows we know least about. Instead the surviving arms keep their relative proportions.
    """
    base = np.clip(np.asarray(base_probs, dtype=float), eps, 1.0 - eps)
    numerator = base_weight * np.log(base / (1.0 - base))
    denominator = np.full(len(base), float(base_weight), dtype=float)

    for probs, weight in zip(encoder_probs, encoder_weights):
        values = np.asarray(probs, dtype=float)
        have = np.isfinite(values)
        clipped = np.clip(np.where(have, values, 0.5), eps, 1.0 - eps)
        numerator = numerator + np.where(have, weight * np.log(clipped / (1.0 - clipped)), 0.0)
        denominator = denominator + np.where(have, float(weight), 0.0)

    if not np.all(denominator > 0):
        raise RuntimeError("ensemble blend has a row with zero total weight")
    return 1.0 / (1.0 + np.exp(-(numerator / denominator)))


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
