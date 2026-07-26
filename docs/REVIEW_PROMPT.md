# Independent code review — prompt for a fresh reviewer

Paste everything below the line into Codex / another agent with repo access.

---

You are reviewing a machine-learning competition codebase with fresh eyes. I need you to find
**correctness bugs**, especially silent ones. Read broadly before concluding, and prefer one
verified real bug over ten speculative ones.

## The situation, honestly

This repo has shipped **four silent-correctness bugs**. Every one produced valid-looking
artifacts, raised no error, and **passed the entire verification suite**. One of them reached
the competition leaderboard and scored *below random* (AUROC 0.4933 against a cross-validated
0.7223). That is the failure mode I need you hunting: **not crashes — plausible wrong answers.**

The four, so you don't re-report them and so you understand the blind spot:

1. **OOF clobbering.** Out-of-fold predictions were keyed by experiment name only, so a
   400-session self-test overwrote the full-data baseline. Every reported "delta vs baseline"
   was then measured against the wrong reference; the headline flipped sign with no error.
   *Both files were valid parquet with correct schemas.*
2. **Model-artifact clobbering.** Same root cause, worse blast radius — a self-test overwrote
   the fold models the submission is built from. Would have shipped a model trained on 1.7% of
   the data. *The files were valid LightGBM boosters either way.*
3. **`main.py` computed 110 of 185 expected features.** Three feature blocks were never wired
   into the inference path. The missing 40% arrived as NaN, which LightGBM silently accepts.
   *Output was correctly formatted and in range.*
4. **Feature-order permutation.** The submission bundle read its column list from
   `importance.parquet`, which is sorted by gain — a permutation of training order in 179 of
   181 positions. `main.py` reordered columns to match; LightGBM reads DataFrames
   positionally; every feature was scrambled. *All 20 verify checks passed. This one cost a
   real leaderboard submission.*

**Pattern:** verification had drifted toward checking *structure* (file exists, columns
present, rows aligned, values in range) and never asked whether the artifact *works*. Assume
more bugs of this shape exist. Assume my abstractions are load-bearing in ways I haven't tested.

## Competition context (constraints are real; some risk disqualification)

- **Task:** predict P(student answers next question correctly) from a K-12 tutoring transcript
  plus a learning-objective description. Metric: **log loss**. ~35k training rows / 22.8k
  sessions; ~10.5k test rows.
- **Submission is code, not predictions.** A zip with `main.py` at the root runs in an offline
  container (Python 3.12, CUDA 12.9, 1×A100, **no network**, 6h cap, logs capped at 500 lines).
- **Disqualification risks:**
  - `main.py` must never print/log anything derived from test data — no excerpts, no counts,
    no aggregates. Only static strings.
  - Each test sample must be processed **independently**. Organizer ruling: *"the only input
    to your model drawn from the test should be that sample's metadata and transcript."*
    Features derived by grouping test rows are **prohibited** (we have a config flag enforcing
    this — verify it cannot leak).
  - External models/data must be openly licensed, **including for development**.
- **Only 3 submissions/week.** A wasted slot is expensive; a wrong answer that *looks* right is
  the worst outcome.

## Architecture

```
src/traceace/            ~10k LOC, 52 files
  tasks.py               task registry: tier guard, manifests, budget ledger. All work
                         runs through tasks.run(name, **kw).
  config/paths/runtime/budget/staging    infra; paths.py has a Drive-FUSE guard
  cv.py                  StratifiedGroupKFold grouped by session_id (NEVER response_id —
                         one session yields up to 10 rows sharing a transcript)
  features/              structural, linguistic, temporal, lo_alignment, feedback,
                         trajectory (+lo_position), embeddings, content, assemble
  models/                baseline (prior, lo_only), gbdt (LightGBM), move_classifier
  evaluate.py            metrics, OOF persistence, delta-vs-baseline guard
  repeated.py            repeated-seed CV: paired deltas, mean ± SD
  experiments.py         evaluate.repeated, interpret.ablation_repeated
  unseen_lo.py           stress test on objectives absent from training
  packaging/
    inference_lib.py     SHARED feature code, copied verbatim into the zip
    main_template.py     the generated main.py
    build_submission.py  assembles the zip
    verify.py            23 checks
tests/                   53 tests, 4 files
docs/                    13 files; STATE.md and DECISIONS.md (13 ADRs) explain the "why"
```

**Read `docs/STATE.md` and `docs/DECISIONS.md` first** — they carry the reasoning and the
history of every bug above.

## Two live suspicions I found while writing this — start here

**A. `lo_alignment` is implemented TWICE with no parity test.**
`features/lo_alignment.py::alignment_features` and
`packaging/inference_lib.py::lo_alignment_features` are separate implementations, each with
its own `_window_dialogue_features`. Parity tests exist for structural, linguistic, temporal,
feedback, trajectory, window-construction and timestamp parsing — **but not for
lo_alignment**. That is precisely the shape of bug #4, unverified, in a block contributing 8
shipped features. Please diff them line by line and test them on identical input.

**B. Every feature block hardcodes `VERSION = "v1"` and caches to parquet keyed on it.**
If someone edits a feature computation without bumping `VERSION`, `cache.load_or_compute`
silently serves the stale cache. There is no content hash. How much has this already bitten
us? Is there a cheap fix (hash the source of the compute function into the key)?

## Highest-risk areas, in priority order

1. **Train/serve parity.** `features/*` (training) vs `packaging/inference_lib.py`
   (inference). Any divergence is silent and catastrophic. Which blocks are genuinely shared
   versus duplicated? Do the parity tests cover every block and every feature, or just a
   sample?
2. **`main.py` correctness** (generated from `main_template.py`). It has several
   `try/except: pass` blocks around feature computation — a swallowed exception yields NaN
   features and confident garbage. Should those fail loudly instead? What happens on a
   malformed/empty transcript?
3. **Leakage.** `cv.py` grouping; `models/gbdt.py::_fold_safe_lo_encoding` (target encoding
   with an inner fold loop — check it carefully); `calibration.py`'s inner cross-fold loop;
   `unseen_lo.py`'s session-preserving split. Any path where a label influences its own
   prediction.
4. **Artifact namespacing.** Bugs 1 and 2 were both this. Is every cache/OOF/model path
   namespaced by everything that varies (subsample, cv_seed, block set, config flags)? Find
   the next collision before it happens.
5. **The rules boundary.** `features.allow_cross_row_aggregates` defaults false and
   `verify.py` enforces it. Can a cross-row value reach the model by any other route? Does
   `main.py` derive anything from more than the row being scored?
6. **Statistical validity.** The paired-delta machinery in `repeated.py` and
   `experiments.py`. The measured noise floor is ~5e-4 on paired log-loss differences; several
   reported effects are near it. Is the pairing correct? Are the intervals honest?

## What a useful finding looks like

- **A concrete failure scenario**: specific input → specific wrong output. Not "this could be
  fragile."
- **Verified where possible.** You can run things (see below). A reproduction beats an
  argument.
- **Ranked by blast radius**: does it corrupt a submission, invalidate a reported number, or
  merely offend taste? Say which.

Please also flag **anything that looks over-engineered or unnecessary**. This grew fast under
time pressure and I would rather delete code than maintain it.

## Please don't spend time on

- Style, formatting, type annotations — `ruff` and `mypy` are clean and enforced in CI.
- Suggesting more features or model architectures. I want *correctness*, not ideas.
- The documentation prose (unless a doc contradicts the code — that IS worth reporting).
- Re-reporting the four bugs above.

## How to run things

```bash
export PATH="$HOME/.local/bin:$PATH"
.venv/bin/pytest -q                    # 53 tests
.venv/bin/ruff check . && .venv/bin/mypy
.venv/bin/python -c "import traceace; traceace.configure(repo_dir='.', quiet=True); \
    traceace.tasks.run('selftest.all')"   # full pipeline on real data, ~25s
```

The real competition data is present locally under `data/raw/` (gitignored, never
redistributable). Feature caches are in `data/features/`, models in `artifacts/models/`.

**Note on the data:** competition rules prohibit transmitting it to third parties. Work from
aggregates — counts, rates, distributions. Do not print verbatim transcript or
learning-objective text.

## Deliverable

A ranked list of findings. For each: what breaks, how you verified it, blast radius, and a
proposed patch. **If you can produce working patches for the high-confidence ones, do.** Run
the test suite before and after.

If you find nothing serious in an area, say so explicitly — that is genuinely useful, because
it tells me where not to keep looking.
