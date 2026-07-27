# Independent code review — ROUND 2

Paste everything below the line into a fresh reviewer with repo access.
(Round 1 ran 2026-07-27, found 8 defects, and was cut off mid-validation. Its findings are
already merged; this brief reflects the post-fix state.)

---

You are reviewing a machine-learning competition codebase with fresh eyes. Find **correctness
bugs**, especially silent ones, then propose and implement **improvements**. Read broadly
before concluding. One verified real bug is worth more than ten speculative ones.

## Track record — this is the important context

This repo has shipped **twelve** distinct correctness defects. Every one produced valid-looking
artifacts, raised no error, and passed the verification suite in force at the time. One reached
the competition leaderboard and scored **below random** (AUROC 0.4933 against a cross-validated
0.7223). **This is the failure class to hunt: not crashes — plausible wrong answers.**

**Found by the author (after damage):**
1. OOF frames keyed by name only → a 400-session self-test overwrote the full-data baseline;
   every reported delta was measured against the wrong reference, headline flipped sign.
2. Same for model artifacts → a self-test could overwrite the fold models a submission is built
   from (would have shipped a model trained on 1.7% of the data).
3. `main.py` computed 110 of 185 features; the rest arrived as NaN, silently accepted.
4. **Feature-order permutation** — the bundle read its column list from `importance.parquet`,
   which is *sorted by gain*: a permutation in 179/181 positions. LightGBM reads DataFrames
   positionally. Every feature scrambled. **All 20 verify checks passed. Cost a real slot.**

**Found by Round 1 review (never caught internally at all):**
5. **Target-encoding leakage** — `_fold_safe_lo_encoding` wrote one shared array across outer
   folds, so each fold overwrote the previous fold's leak-free encoding with one derived from
   data including its own labels. Flipping *only* fold 0's validation labels moved fold 0's own
   encoding by −0.25. Inflated every CV number in the repo.
6. **Subsample cohort mismatch** — `cv.build` selected "first N sessions" in *train_features row
   order*; session-level blocks used *transcript filename order*. At N=400 they overlapped in
   **9 sessions (2.2%)**. ~98% of self-test rows had all-NaN features. `selftest.all` passed for
   weeks while testing essentially nothing.
7. Verifier's prediction check inspected a hardcoded archive, not the caller's.
8. Stale calibrator survived when "no calibration" won.
9. Cross-row rules guard could return PASS when the flag was enabled.
10. Unreadable transcripts produced tree predictions instead of the documented prior fallback.
11. OOF frames of equal length but different cohorts could be compared.
12. `ensemble.blend` accepted an implicit experiment list.

**The pattern:** verification checked *structure* (file exists, columns present, rows aligned,
values in range) and never asked whether the artifact *works*. Assume more of this shape exists.

**A diagnostic that was missed for weeks, now enforced:** constant-0.5 predictions score
ln(2) = 0.6931 on any binary labels. Two container smoke runs scored 0.8543 and 0.8330 —
*worse than a coin flip*, i.e. confidently wrong — and were dismissed as "fake data,
meaningless." That one line of arithmetic would have caught bug #4 before it cost a submission.

## Competition context (some constraints risk disqualification)

- **Task:** predict P(student answers next question correctly) from a K-12 tutoring transcript
  plus a learning-objective description. Metric **log loss**. 35,072 train rows / 22,821
  sessions; 10,508 test rows. One session yields up to 10 rows **sharing a transcript** — so CV
  groups by `session_id`, never `response_id`.
- **Submission is code.** A zip with `main.py` at root runs offline (Python 3.12, CUDA 12.9,
  1×A100, **no network**, 6h cap, logs capped 500 lines × 500 chars).
- **Disqualification risks:**
  - `main.py` must never print/log anything derived from test data — no excerpts, no counts,
    no aggregates. Static strings only.
  - Each test sample processed **independently**. Organizer ruling: *"the only input to your
    model drawn from the test should be that sample's metadata and transcript."* Features
    derived by grouping test rows are **prohibited** (config flag + verify check enforce this).
  - External models/data must be openly licensed **including for development**; competition
    data must never be uploaded to an API.
- **3 submissions/week**, ~2 remaining this week. Current leaderboard: rank #229 from the
  broken submission; a fixed one is pending.

## Architecture

```
src/traceace/            ~10.8k LOC, 53 files
  tasks.py               registry: tier guard, manifests, budget ledger. All work runs
                         through tasks.run(name, **kw).
  config/paths/runtime/budget/staging   infra; paths.py guards against iterating a Drive mount
  cv.py                  StratifiedGroupKFold grouped by session_id
  features/              structural, linguistic, temporal, lo_alignment, feedback,
                         trajectory(+lo_position), embeddings, window_embeddings, content,
                         assemble, common
  models/                baseline (prior, lo_only), gbdt (LightGBM), move_classifier
  evaluate.py            metrics, OOF persistence, cohort-compatibility checks
  repeated.py            repeated-seed CV: paired deltas, mean ± SD
  experiments.py         evaluate.repeated, interpret.ablation_repeated
  unseen_lo.py           stress test on objectives absent from training
  annotate.py            tutoring-move annotation (heuristic + vLLM backends)
  packaging/
    inference_lib.py     SHARED feature code, copied verbatim into the zip
    main_template.py     the generated main.py
    build_submission.py  assembles the zip
    verify.py            24 checks
tests/                   66 tests, 5 files
docs/                    15 files. STATE.md = status; DECISIONS.md = 15 ADRs explaining every
                         bug above and why each fix is shaped as it is.
```

**Read `docs/STATE.md` and `docs/DECISIONS.md` first.** ADR-013/014/015 cover the four most
recent defects in detail.

## Known blind spots — verified gaps, start here

These are measured, not guessed:

**A. Modules with zero test coverage:** `features/window_embeddings.py`, `models/move_classifier.py`.

**B. Tasks that have never once been executed:** `ensemble.blend`, `features.embeddings`,
`maintenance.sync_artifacts`. Code that has never run is code that has never been right.

**C. The GPU path is entirely unexercised at scale.** `features/window_embeddings.py` +
`features/content.py` (pooled embeddings → PCA) were validated only on 40 sessions on CPU. They
are staged for an L4 run that has not happened. Review them as if they are about to run and cost
real money.

**D. Determinism is never tested.** Nothing asserts that running the same task twice produces
identical artifacts. Given how many bugs here were state/ordering related, this seems worth a
test.

**E. `docs/` may now contradict the code.** Several ADRs were written before their own fixes were
revised. A doc that misstates behaviour is a real defect — it is how bug #5 hid (an accurate
docstring describing an implementation that did something else).

**F. `interpret.ablation_repeated` and `unseen_lo` results in FINDINGS.md were all measured
under the leakage of bug #5** and have not been re-run. Are the harnesses themselves correct
now?

## Highest-risk areas, in priority order

1. **Train/serve parity.** `features/*` vs `packaging/inference_lib.py`. `lo_alignment` now
   delegates to the shipped implementation; check the others genuinely share code rather than
   merely agreeing today. Do parity tests cover every block *and every feature*, or a sample?
2. **`main.py`** (generated from `main_template.py`). It has `try/except` blocks around feature
   computation. Which swallow errors, and should they? What happens with an empty transcript, a
   malformed CSV, a session file that is missing entirely, a duplicate `response_id`?
3. **Leakage.** `models/gbdt.py::_fold_safe_lo_encoding` (recently rewritten — verify the fix is
   actually correct, don't take the ADR's word), `calibration.py`'s inner loop, `unseen_lo.py`'s
   session-preserving split, `cv.py`. Any path where a label influences its own prediction.
4. **Artifact identity.** Bugs 1, 2 and 6 were all "two things that should mean the same thing
   didn't." Caches now carry a source hash; OOF and model dirs are namespaced by
   subsample/cv_seed. Find the next collision *before* it happens. Is anything still keyed on
   something that can vary silently — block set, config flags, feature-order?
5. **The rules boundary.** Can a cross-row value reach the model by any route? Does `main.py`
   derive anything from more than the row being scored? Does anything print test-derived data?
6. **Statistical validity.** `repeated.py` and `experiments.py`. Measured noise floor on paired
   log-loss deltas is ~5e-4 and several reported effects sit near it. Is the pairing right? Are
   the intervals honest? Is the unseen-LO split sound?

## Also wanted: improvements, not just bug fixes

The brief for round 1 asked only for correctness. This time also propose and implement:

- **Deletions.** This grew fast under time pressure. What is unused, over-abstracted, or
  earning less than it costs to maintain? I would rather delete code than keep it.
- **Speed.** Feature builds take ~20 min over 22.8k sessions; `selftest.all` takes 75 s. Both
  are on the critical path of every iteration. Profile before optimizing.
- **Making the remaining silent paths loud.** Anywhere a wrong answer is possible without an
  error, propose the assertion that would surface it.
- **Test quality.** 66 tests exist and they missed 8 defects. Which are load-bearing and which
  are theatre? Mutation-style tests (change an input that shouldn't matter, assert the output
  doesn't move) proved far more effective here than assertion-count.

## What a useful finding looks like

- **A concrete failure scenario**: specific input → specific wrong output. Not "this is fragile."
- **Verified.** You can run everything (below). A reproduction beats an argument.
- **Ranked by blast radius**: corrupts a submission / invalidates a reported number / offends
  taste. Say which.
- **If you can produce a working patch, do**, and run the suite before and after.

Explicitly say where you looked and found **nothing** — that tells me where to stop looking.

## Please don't spend time on

- Style, formatting, type annotations — `ruff` and `mypy` are clean and CI-enforced.
- Proposing new features or model architectures. Correctness and simplification, not ideas.
- Re-reporting the twelve defects above.

## How to run things

```bash
export PATH="$HOME/.local/bin:$PATH"
.venv/bin/pytest -q                                    # 66 tests
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
.venv/bin/python -c "import traceace; traceace.configure(repo_dir='.', quiet=True); \
    traceace.tasks.run('selftest.all')"                # end-to-end on real data, ~75s
.venv/bin/python -c "import traceace; traceace.configure(repo_dir='.', quiet=True); \
    traceace.tasks.run('submission.verify', smoke=True)"   # 24 checks incl. prediction sanity
```

Real competition data is present under `data/raw/` (gitignored, non-redistributable). Feature
caches in `data/features/`, models in `artifacts/models/`, run manifests in `runs/`.

**Data handling:** competition rules prohibit transmitting the data to third parties. Work from
aggregates — counts, rates, distributions. **Do not print verbatim transcript or
learning-objective text** into logs or your context.

## Deliverable

A ranked list of findings, each with: what breaks, how you verified it, blast radius, proposed
patch. Then a short list of improvements with the reasoning for each.

If you run low on budget, **prioritise depth over breadth** — finish and validate what you have
started rather than opening a new track. Round 1 ended mid-validation with an unverified patch
set, which cost real time to check by hand afterwards.
