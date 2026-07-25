# STATE.md — read this first

**Last updated:** 2026-07-25 · **git SHA:** _(initial commit — see `git log -1`)_
**Competition deadline:** model submissions 2026-08-27 23:59 UTC · write-up 2026-09-15
**Entry:** solo · **Units remaining: 733.00 / 733** (everything so far ran on CPU, 0 units)

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
| `model.gbdt` (4 blocks) | 0.54306 | 0.7223 | −0.00878 |
| **`model.gbdt` + Platt (current best)** | **0.543005** | **0.7223** | **−0.00883 ✅ beats the bar** |

5-fold session-grouped CV on all 35,072 responses. Full table: [`EXPERIMENTS.md`](EXPERIMENTS.md).
**No leaderboard submission has been made** — LB score unknown.

## What works
- **All 27 tasks** registered and runnable via `traceace.tasks.run(name)`.
- **Data**: ingest normalizes suffixed filenames by content shape; 22,821 transcripts
  consolidated. Measured facts in [`DATA.md`](DATA.md).
- **CV**: session-grouped folds built once and persisted; leakage test green in CI.
- **Features**: 5 blocks built (4 in the default stack), **149 features**, all cached at
  full scale. Includes the new `feedback` block (tutor corrective feedback, +0.00046).
- **Model**: LightGBM, 5 folds, OOF persisted, cross-fold importance with dispersion.
- **Research artifacts**: figures (PNG+PDF) in `artifacts/figures/`; 5-block ablation done;
  [`FINDINGS.md`](FINDINGS.md) has 7 key findings + 6 negative results (incl. two retractions
  of our own earlier conclusions).
- **Semantic LO-alignment**: `features.window_embeddings` (BAAI/bge-small-en-v1.5, MIT) and
  `lo_alignment(backend="embedding")` implemented and **validated end-to-end on CPU**.
- **Submission**: 1.29 MB zip, `main.py` at root, **all 16 verify checks pass**, smoke run
  4.3 s → **0.126 h projected** for the full test set (cap 6 h).
- **Quality gates**: ruff clean · mypy clean (39 files) · 41 tests pass · `selftest.all`
  green in 15.5 s.

## Known problems / open risks
1. **The margin over the baseline is still thin** (−0.0088). Most of the model's power is the
   topic prior, not the transcript. Improving the *transcript* contribution remains the main
   modeling task — and the main research story.
2. **The blocks are largely substitutes, not complements.** LO-alignment alone (0.54921) and
   structural alone (0.54878) score nearly the same. This caps what any one new block adds
   and is the strongest argument for the semantic-alignment upgrade (next action 1).
3. **Temporal block is excluded** — measured contribution −0.00049 (it hurts). Kept
   registered so the negative result stays reportable. See [`FINDINGS.md`](FINDINGS.md) N1.
4. **Semantic LO-alignment is built but not run at scale.** `features.window_embeddings`
   is validated end-to-end on CPU (40 sessions) but the full extraction needs an L4;
   projected **35–52 min ≈ 3–4 units**. This is the single highest-value pending experiment.
5. **`annotate.moves` has only run with the `heuristic` backend.** The vLLM backend is
   written but unexercised.
6. Local env now has **torch (CPU) + sentence-transformers** installed so GPU code paths can
   be validated before spending units — a deliberate relaxation of ADR-005.
7. Git SHAs in early run manifests read `unknown` (they predate the first commit).

## Next actions, in order
1. **L4 session — semantic LO-alignment (~3–4 units).** THE highest-value experiment.
   `run("features.window_embeddings", subsample=500)` to smoke, then the full run, then
   `run("features.lo_alignment", backend="embedding", force=True)` and re-run `model.gbdt`.
   Lexical matching is the current ceiling: objectives and dialogue rarely share literal
   wording. Projected 35–52 min on L4 (measured from CPU throughput).
2. **Move taxonomy**: `annotate.moves` with the vLLM backend seeded by the feedback
   categories, then `model.move_classifier`, then add move-distribution features.
3. **First real submission** once step 1 lands. Budget: 3/week, ~15 attempts left.
4. **A100 timing validation (~4 units)** only immediately before a real submission.
5. Tuning is **not** a next action — a capacity sweep showed the current config is already
   at the plateau (FINDINGS N5).

## Compute budget
733 units, **0.00 spent**. Plan (ADR-008): ~0 CPU ladder · ~5 embeddings · ~40 LLM
annotation · ~30 classifier iteration · ~15 A100 timing. **~600 held in reserve until
week 3.** Flag anything projected above 25 units for a single task.

## Where the numbers came from
`runs/` holds a manifest per task run (git SHA, seed, wall time, units, metrics) and
`runs/budget.jsonl` is the unit ledger. [`EXPERIMENTS.md`](EXPERIMENTS.md) is regenerated
from them by `docs.build` — never edit it by hand.
