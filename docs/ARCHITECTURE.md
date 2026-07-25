# ARCHITECTURE.md — how the pieces fit, and why

## The shape of the problem (this drives everything)

Three measured facts (see [`DATA.md`](DATA.md)) determine the design:

1. **Sessions are the unit of leakage.** 35,072 responses come from 22,821 sessions; one
   session can yield up to 10 responses sharing an identical transcript. → *Folds group by
   session, always.*
2. **43.3% of label variance is within-session.** Session-level features are mathematically
   incapable of reaching it. → *LO-conditioned features are mandatory, not a bonus.*
3. **Transcripts are ~5.3K tokens (median), not 10–15K.** → *The 6h inference cap is not
   binding; choose architecture on signal quality and dev budget instead.*

## Layered design

```
  conf/base.yaml            all tunables, unit-rate table, seed, cv settings
        │
  config.py ──► paths.py (Drive guard) ──► staging.py (Drive⇄local SSD)
        │            │
        │       runtime.py (tier detect) ──► budget.py (unit ledger)
        │            │
        └──────► tasks.py  ── the ONE entrypoint: registry + tier guard +
                     │         manifest + budget + progress header
                     ▼
   data.* ► eda.* ► cv.build ► features.* ► model.* ► calibrate ► ensemble
                                                │
                                    evaluate.* / interpret.*  ► docs/FINDINGS.md
                                                │
                                    packaging.* ► submission.zip
```

**Everything is a task.** `tasks.run(name, **kw)` is the single way work happens. Each task
declares its tier, is idempotent (loud cache skip, `force=True` to override), writes a
manifest to `runs/<task>/<ts>.json` (git SHA, seed, kwargs, wall time, units, metrics), and
appends to the budget ledger. The notebook is a thin wrapper that git-pulls and calls this.

### Why a tier guard
Colab units burn **while the runtime is connected**, not while it computes — so an idle
attached A100 is the single largest waste vector. Tasks declare `requires` (minimum) and
`max_tier` (maximum). Running a CPU task on an A100 **refuses by default** and names the
runtime to switch to. Discipline you must remember is discipline you will forget.

### Why the Drive guard
Google Drive on Colab is a FUSE mount: ~100–300 ms **per file operation**. Iterating 22,821
transcript CSVs over it would take hours of pure syscall latency. So `paths.iter_files()`
routes all directory walking through `assert_not_drive()`, which raises on a Drive path.
Drive holds a handful of large files; `/content` local SSD is the working disk. Extracting
*to* Drive is impossible by construction.

## Feature blocks

| Block | Key | What it captures | Cost |
|---|---|---|---|
| `structural` | session | turn counts, talk ratio, run lengths, role balance | CPU |
| `linguistic` | session | questioning/hedging/affirmation + **ASR disfluency** per role | CPU |
| `temporal` | session | latency and pacing, **robust statistics** | CPU |
| `lo_alignment` | **response** | **LO-conditioned key moments** — the within-session tie-breaker | CPU |
| `embeddings` | session | frozen ModernBERT vectors, extracted once on L4 | ~5 units |

Blocks are prefixed (`struct_`, `ling_`, `temp_`, `lo_`, `emb_`) so `interpret.ablation` can
drop a whole block by prefix and measure its marginal contribution. `features/assemble.py`
joins session-level blocks by `session_id` and the response-level block by `response_id`.

### The LO-alignment block, specifically
For each (session, LO) pair: slide a window over the utterances → score each window's
cosine relevance to the LO text → pool over the top-k windows. Output includes *where* the
relevant windows sit (`lo_best_pos`, `lo_topk_pos_spread`) and what the dialogue looks like
*inside* them (`lo_kw_student_talk_ratio`, `lo_kw_hedge_rate`, …). Those inner statistics
are what actually differ between two objectives in one session.

The vectorizer is fit on **training LO text only** — deterministic, and never touched by
the test set, satisfying the "identical parameters with absent test data" rule.

Two backends: `lexical` (TF-IDF, CPU, default — keeps the self-test GPU-free) and
`embedding` (dense, L4). The lexical one runs everywhere so the pipeline is never blocked.

## Modeling ladder (cheap-first, by design)

```
baseline.prior      global base rate            → the floor      (logloss ≈ 0.609)
baseline.lo_only    per-LO mean, no transcript  → THE BAR        (the anti-goal)
model.gbdt          LightGBM on feature blocks  → the workhorse  (CPU, 0 units)
features.embeddings frozen encoder + GBDT       → if needed      (~5 units, once)
model.encoder       fine-tuning                 → only if the ladder plateaus
```

Every report prints `delta_vs_lo_only`. A model that cannot beat `lo_only` has learned
which topics are hard, not what makes tutoring work — the organizers' explicit anti-goal.

**Calibration is not an afterthought.** Log loss responds more to calibration than to
ranking, so `calibrate.fit` compares none/Platt/isotonic with an *inner* cross-fold loop
(so the reported gain is itself honest) and keeps the winner.

## Research artifacts as a first-class output

~70% of the judged component rewards transferable insight, so `interpret.report` runs
alongside every model and emits cross-fold importance with dispersion, per-slice
performance, reliability diagrams, key-moment position distributions, and move-taxonomy
relationships — all as publication-quality PNG+PDF sized for the 4-page write-up.
`interpret.ablation` gives each block's marginal contribution so claims are about *what*
mattered.

## The submission path

```
model.gbdt ──► submission.build ──► submission.zip
                                     ├── main.py          (generated, root level)
                                     ├── inference_lib.py (copied VERBATIM)
                                     └── assets/model.joblib
                                              │
                            submission.smoke ─┴─► submission.verify
```

**One feature implementation, two callers.** `packaging/inference_lib.py` is imported by
the training pipeline *and* copied verbatim into the zip, so train/serve skew cannot
happen; `tests/test_inference_parity.py` asserts value-for-value equality. `inference_lib`
may not import `traceace` and may not print — both enforced by tests.

`submission.smoke` runs `main.py` as a **subprocess** in a container-shaped directory —
catching import failures, path assumptions and stray stdout that an in-process call would
mask. `submission.verify` is deliberately paranoid because only ~15 real attempts remain:
it fails on main.py placement, row-order mismatch, out-of-range probabilities, non-literal
log emissions (AST scan), network-capable imports, undisabled progress bars, and runtime
projections above 4.5 h.

## Where state lives

| Kind | Location | In git? |
|---|---|---|
| Raw + interim data, features | `data/` | ❌ never |
| Models, OOF, figures, logs | `artifacts/` | ❌ never |
| Run manifests, budget ledger | `runs/` | ❌ never |
| Code, config, docs, notebook | `src/`, `conf/`, `docs/`, `notebooks/` | ✅ |

`.gitignore` plus a native pre-commit hook (blocking >1 MB files and data patterns) enforce
this. GitHub is the source of truth; Colab is disposable; Drive holds the large artifacts.
