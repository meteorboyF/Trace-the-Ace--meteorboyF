# STATE.md — read this first

**Last updated:** 2026-07-25 · **git SHA:** _(initial commit — see `git log -1`)_
**Competition deadline:** model submissions 2026-08-27 23:59 UTC · write-up 2026-09-15
**Entry:** solo · **Units remaining: 733.00 / 733** (nothing has run on a paid runtime yet)

> New session? Read [`../CLAUDE.md`](../CLAUDE.md) → [`BRIEF.md`](BRIEF.md) → this file.
> Then [`DATA.md`](DATA.md) for measured facts and [`RUNBOOK.md`](RUNBOOK.md) for recipes.

---

## Status: pipeline complete and verified end-to-end. No submission made yet.

The full cheap ladder is built, runs on the real data, and produces a valid `submission.zip`.
Everything so far has run on **CPU for zero units**.

## Best score

| Model | CV log loss | AUC | Δ vs `lo_only` |
|---|---|---|---|
| `baseline.prior` | 0.60876 | 0.500 | +0.0569 |
| `baseline.lo_only` (the bar) | 0.55184 | 0.707 | — |
| **`model.gbdt` (current best)** | **0.54401** | **0.7208** | **−0.00783 ✅ beats the bar** |

5-fold session-grouped CV on all 35,072 responses. Full table: [`EXPERIMENTS.md`](EXPERIMENTS.md).
**No leaderboard submission has been made** — LB score unknown.

## What works
- **All 27 tasks** registered and runnable via `traceace.tasks.run(name)`.
- **Data**: ingest normalizes suffixed filenames by content shape; 22,821 transcripts
  consolidated. Measured facts in [`DATA.md`](DATA.md).
- **CV**: session-grouped folds built once and persisted; leakage test green in CI.
- **Features**: 4 blocks, 127 features, all cached at full scale to `data/features/`.
- **Model**: LightGBM, 5 folds, OOF persisted, cross-fold importance with dispersion.
- **Research artifacts**: 16 figures (PNG+PDF) in `artifacts/figures/`; ablation done;
  [`FINDINGS.md`](FINDINGS.md) has 6 key findings + 4 negative results.
- **Submission**: 1.29 MB zip, `main.py` at root, **all 16 verify checks pass**, smoke run
  4.3 s → **0.126 h projected** for the full test set (cap 6 h).
- **Quality gates**: ruff clean · mypy clean (39 files) · 41 tests pass · `selftest.all`
  green in 15.5 s.

## Known problems / open risks
1. **The margin over the baseline is thin** (−0.0078). Most of the model's power is the
   topic prior, not the transcript. Improving the *transcript* contribution is the main
   modeling task — and the main research story.
2. **Structural features contribute nothing** (ablation: −0.00004). Candidate for removal;
   kept for now because they are cheap and interpretable. See [`FINDINGS.md`](FINDINGS.md) N1.
3. **Calibration does not help** (raw beats Platt and isotonic). Re-check after any model
   change rather than assuming. See N2.
4. **Embeddings not yet extracted** — the L4 path is written and cached-by-design but has
   never run. This is the largest untested code path.
5. **`annotate.moves` has only run with the `heuristic` backend.** The vLLM backend is
   written but unexercised.
6. Local env is **CPU-only by design** (ADR-005) — torch/transformers are not installed
   locally, so GPU code paths cannot be smoke-tested here.
7. Git SHAs in early run manifests read `unknown` (they predate the first commit).

## Next actions, in order
1. **Push to GitHub** and confirm CI goes green.
2. **Colab CPU session**: re-run `data.*` → `features.*` → `model.gbdt` to confirm parity
   with local results (~0 units).
3. **Improve the transcript signal** — this is where both the score and the paper live:
   - LO-alignment currently uses TF-IDF; try the embedding backend for window relevance.
   - Add move-distribution features from `model.move_classifier` into the design matrix.
   - Tune LightGBM (current params are sensible defaults, untuned).
4. **L4 session (~5 units)**: `features.embeddings`, smoke at `subsample=500` first.
   Cached forever afterwards.
5. **First real submission** once step 3 shows a clear gain. Budget: 3/week, ~15 left.
6. **A100 timing validation (~4 units)** only immediately before a real submission.

## Compute budget
733 units, **0.00 spent**. Plan (ADR-008): ~0 CPU ladder · ~5 embeddings · ~40 LLM
annotation · ~30 classifier iteration · ~15 A100 timing. **~600 held in reserve until
week 3.** Flag anything projected above 25 units for a single task.

## Where the numbers came from
`runs/` holds a manifest per task run (git SHA, seed, wall time, units, metrics) and
`runs/budget.jsonl` is the unit ledger. [`EXPERIMENTS.md`](EXPERIMENTS.md) is regenerated
from them by `docs.build` — never edit it by hand.
