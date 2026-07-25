# EXTERNAL_ASSETS.md — models, datasets, and runtime package parity

Competition rule: external datasets and pretrained models are permitted **only if publicly
available and openly licensed for commercial use**. Winning solutions must be
MIT-licensable. No NC / CC BY-NC / research-only terms.

---

## 1. Runtime package parity (§7 check 5)

Source of truth: `drivendataorg/tutoring-outcomes-runtime` → `runtime/pyproject.toml`
(read 2026-07-25). Runtime requires `>=3.12,<3.13`.

**Official runtime dependency list (verbatim):** accelerate, aiofiles, aiohttp, attrs,
certifi, chromadb, click, cloudpickle, cupy-cuda12x, datasets, dill, diskcache, einops,
fastapi, filelock, fsspec, gensim, grpcio, httpcore, httpx, huggingface-hub,
importlib-metadata, jinja2, jsonpickle, langchain-community, langchain, **lightgbm**,
loguru, more-itertools, ms-swift, ninja, numba, **numpy**, opencv-python-headless,
packaging, **pandas**, peft<0.19, pillow, polars, psutil, pydantic-settings, pydantic,
pytorch-lightning, pyyaml, regex, requests, sacremoses, safetensors, **scikit-learn**,
**scipy**, sentence-transformers, sentencepiece, spacy, starlette, statsmodels, thinc,
tiktoken, timm, tokenizers, torch==2.11.0+cu129, torchaudio==2.11.0+cu129,
torchvision==0.26.0+cu129, tqdm, transformers, urllib3, uvicorn, vllm, xarray, xformers,
catboost. Test extra: pytest>=9.0.

### Reconciliation against what the **submission** imports

| Package | In runtime? | Used at inference? | Verdict |
|---|---|---|---|
| numpy | ✅ | ✅ | OK |
| pandas | ✅ | ✅ | OK |
| scikit-learn | ✅ | ✅ (TF-IDF vectorizer, calibrator) | OK |
| lightgbm | ✅ | ✅ (boosters) | OK |
| scipy | ✅ | transitively via sklearn | OK |
| joblib | via scikit-learn | ✅ (asset loading) | OK — bundled with sklearn |

**Result: zero package-addition requests needed.** The submission's entire import surface
is present in the official image. No GitHub issue required.

### Dev/Colab-only packages (NEVER imported at inference)

| Package | In runtime? | Why we use it | Risk |
|---|---|---|---|
| matplotlib | ❌ | figures for `interpret.report` | none — dev only |
| zstandard | ❌ | `tar.zst` rollups for Drive sync | none — dev only |
| pyarrow | transitive (via `datasets`) | parquet caches | none — submission never reads parquet |
| tiktoken | ✅ | exact token counts in `eda.transcripts` | none — dev only |

> **Standing rule.** Before adding any import to `packaging/inference_lib.py` or the
> generated `main.py`, check it against the table above. `submission.verify` statically
> scans for network-capable imports but cannot know whether a package exists in the image.

---

## 2. Pretrained models

| Model | License | Commercial use | Runtime availability | Status |
|---|---|---|---|---|
| `answerdotai/ModernBERT-base` | **Apache-2.0** | ✅ permitted | loadable via `sentence-transformers`/`transformers` (both in runtime); weights must be **vendored** into the zip (no network) | **Approved** — default for `features.embeddings` (8192 ctx covers ~99% of sessions) |
| `microsoft/deberta-v3-base` | **MIT** | ✅ permitted | same | Approved alternative |
| `sentence-transformers/all-MiniLM-L6-v2` | **Apache-2.0** | ✅ permitted | same | Approved — cheap baseline encoder |
| `Qwen/Qwen2.5-7B-Instruct` | **Apache-2.0** | ✅ permitted | vLLM in runtime | Approved **dev-time only** (`annotate.moves`); by ADR-004 it never enters the submission |
| Llama-family | bespoke community licence | ⚠️ conditional | — | **Flagged** — do not build on it without a forum question first |
| Any `-NC` / CC BY-NC / research-only asset | — | ❌ | — | **Prohibited** |

**Currently in the submission path: none.** The shipped model is LightGBM plus a TF-IDF
vectorizer, both trained by us on competition data only. Nothing external is vendored yet.

---

## 3. External datasets

None used. If one is ever added, record here: name, URL, licence, commercial-use verdict,
and how it enters the pipeline.

---

## 4. Vendoring checklist (before any model enters the submission)

1. Confirm the licence permits commercial use, and record it above.
2. Download weights locally and place them under `submission/assets/` (gitignored).
3. Confirm the loader works **offline** (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
4. Confirm the zip stays under 55 GB (`submission.verify` enforces this).
5. Re-run `submission.smoke` and confirm the projected full runtime stays under 4.5 h.
