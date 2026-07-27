# Independent review — ROUND 3

> Supersedes the round-1 and round-2 briefs (in git history). Round 1 found 8 defects,
> round 2 found 4. Both are merged; this brief reflects the post-fix state.
> Paste everything below the line into a fresh reviewer with repo access.

---

You are reviewing a solo competition entry for DrivenData's **"Trace the Ace"**: predict
P(student answers the next question correctly) from a tutoring-session transcript plus a
learning objective. Metric is **log loss**. Repo `TraceTheAce`, branch `main`, Python 3.12.

Your job, in priority order:

1. **Find correctness bugs** — especially *silent* ones that produce plausible numbers while
   being wrong.
2. **Fix them**, each with a regression test that fails before and passes after.
3. **Improve the codebase** where it is genuinely weak.
4. **Make the notebook correct and runnable**, because a paid GPU session is about to be run
   through it and it is currently the weakest artifact in the repo.

---

## ⛔ Hard rules — breaking these is worse than any bug you could find

**Competition data must never leave the machine and must never be displayed.** Participants
agree not to "transmit, duplicate, publish, redistribute or otherwise provide or make
available the Data to any party not participating in the Competition", and "no competition
data should be uploaded to an API."

Practically, for you:

- **Never print, quote, echo, or paste verbatim transcript content, learning-objective text,
  or student/tutor utterances** — not in your reasoning, not in your report, not into a
  scratch file. Work from **aggregates**: counts, rates, distributions, correlations. If you
  need an example to reason about a bug, compute the **statistic**, not the text.
- **Never commit data.** `.gitignore` plus `.git/hooks/pre-commit` block it. Data lives in
  `data/raw/` and on Drive, never in git. If you touch those patterns, verify the hook still
  blocks a `.csv` *and* that test files are still tracked — a previous edit made CI silently
  collect **zero tests** for days.

**Never tune against leaderboard feedback.** 3 submissions/week; the leaderboard is not a
validation set.

**Never claim a gain without repeated-seed evidence.** At 35K rows the paired per-fold delta
SD is ≈5e-4 and the headline moves 0.00105 across fold assignments. Any single-seed
improvement below ~1e-3 is noise. Use `interpret.ablation_repeated` / `evaluate.repeated`,
which report mean ± SD.

---

## Orientation

```
src/traceace/
  tasks.py               registry + tier guard; run via traceace.tasks.run(name, **kw)
  cv.py                  StratifiedGroupKFold(groups=session_id)  ← never response_id
  features/              assemble.py (block registry), structural, linguistic, temporal,
                         feedback, trajectory, lo_position, lo_alignment, embeddings, content
  models/gbdt.py         LightGBM, 5 folds, fold-safe target encoding
  unseen_lo.py           evaluate.unseen_lo + calibrate.shrinkage
  packaging/
    build_submission.py  writes submission.zip
    inference_lib.py     SHARED train/serve feature code — copied VERBATIM into the zip
    main_template.py     the main.py that ships
    verify.py            24 checks
conf/base.yaml           paths, seed, cv, unit-rate table (never hardcode rates)
notebooks/Trace_the_Ace_Runner.ipynb     the Colab wrapper  ← WEAKEST ARTIFACT
docs/                    BRIEF.md (spec), STATE.md (status), DATA.md (measured facts),
                         DECISIONS.md (ADR-001..018), FINDINGS.md (paper draft)
```

```bash
.venv/bin/pytest                                    # 74 tests, ~1.4 s
.venv/bin/ruff check . && .venv/bin/mypy
```

Every task requires `traceace.configure(repo_dir=".")` first or it raises.

Read `docs/BRIEF.md` (authoritative spec), then `docs/STATE.md`, then `docs/DATA.md` for
measured data facts, before touching features or models.

---

## Priority 0 — the notebook (this is why you were called)

`CLAUDE.md` calls the notebook "a thin stable wrapper" around the package. It has **drifted
from the package** and is about to drive a paid GPU run. A measured diff of its `run("...")`
calls against the task registry shows **9 of 38 tasks unreachable**, and the gaps are not
cosmetic:

- **Cell 12 builds `structural, linguistic, temporal, lo_alignment`.** It omits
  **`features.feedback` and `features.trajectory`**. Trajectory is the single strongest block
  (**+0.00226 ± 0.00011**). Meanwhile `lo_alignment` measures **−0.00030 ± 0.00024** — it
  *hurts* — and is still built. A clean notebook run therefore trains on the wrong block set.
- **Cell 26 never calls `calibrate.shrinkage`.** Deployment shrinkage (ADR-017) is worth
  **+0.00715**, an order of magnitude more than any feature block. A notebook-driven rebuild
  silently ships a submission without it, undoing the largest single gain in the project.
- Also unreachable: `evaluate.unseen_lo`, `selftest.all`, `features.feedback`,
  `features.trajectory`, `features.embeddings`, `eda.roles`, `eda.lo_conditioning`,
  `ensemble.blend`.
- Cell 26 calls `run("submission.verify", smoke=True)`. Check `smoke=True` still means what
  the cell author intended after ADR-018 changed how the format file is resolved.

**What to do.** Make the notebook produce *exactly* the pipeline the package considers
current, and make that property **enforced by a test rather than by care** — e.g. parse the
notebook's `run(...)` calls and assert the set matches an explicit declared manifest, so
adding a task without wiring it in fails CI. A wrapper that can drift silently *will* drift
again; this is the second time one has silently dropped features, and the first cost a
submission.

Then confirm the notebook is safe to run top-to-bottom on a fresh Colab: cell ordering, the
Drive-mount/clone/`sync()` cell, tier annotations matching each task's real `requires=`, and
GPU cells marked so an operator cannot leave an L4 attached and idle through the CPU cells.
Idle attached GPUs are the project's #1 unit waste; the budget is 733 units with ~733 left.

---

## Priority 1 — the GPU path, which has never executed at scale

`features.window_embeddings` / `features.embeddings` (BAAI/bge-small-en-v1.5, MIT) and
`features.content` (pooled top-k window vectors → PCA-48) are **validated on CPU at 40–500
sessions and never run at full scale**. Real units are about to be spent on them.

Audit for what only breaks at scale or only on GPU:

- batching and OOM behaviour; whether a mid-run failure loses all completed work
  (is there resumability, or does an 8-hour run restart from zero?)
- **producer/consumer cache-path agreement.** This has already broken once: a source-hash
  change made `features.window_embeddings` write to one path while
  `lo_alignment(backend="embedding")` read another, so a *paid* extraction would have
  reported "missing". Re-verify by **executing both**, not by reading.
- dtype/device assumptions, `.cpu()`/`.numpy()` transfers, deterministic seeding
- **PCA fold discipline** in `features.content`. If PCA is fit across all rows including
  validation folds, it leaks — the exact shape of the target-encoding bug already shipped
  here. Check this specifically.
- `annotate.moves` with `backend="vllm"` is written but **never executed** (~40 units). It
  must run fully locally; competition data may not be sent to any API.

---

## Priority 2 — general bug hunt

Assume the model is subtly broken until proven otherwise. **Thirteen silent-correctness
defects have already been found here.** Four reached a submission; one scored *below random*
on the leaderboard. Every one was live while the test suite was green — so a passing suite is
weak evidence, not a clean bill of health.

### Already found and fixed — don't re-report, but *do* check the fixes hold

| # | Defect | ADR |
|---|---|---|
| 1 | `.gitignore` `test_*` swallowed all test files; CI collected 0 tests | — |
| 2 | pandas-3 pyarrow `cumsum` on booleans | — |
| 3 | OOF clobbering: subsampled runs overwrote full-data baselines, flipping the reported delta's sign | ADR-015 |
| 4 | Model-artifact clobbering: selftest could overwrite submission fold models | ADR-015 |
| 5 | `main.py` produced 110 of 181 features — feedback/trajectory/lo_position never wired in, arriving as NaN | ADR-010 |
| 6 | **Feature-order permutation**: bundle took column order from `importance.parquet` (sorted by gain), permuting 179/181 positions; LightGBM reads positionally. Shipped: LB 0.8006, AUROC 0.4933, rank #229 | ADR-013 |
| 7 | sklearn 1.9.0 local vs 1.8.0 in the container | ADR-012 |
| 8 | **Target-encoding leakage**: one `enc` array shared across outer folds; flipping fold 0's validation labels moved its own encoding by −0.25. Inflated every CV number before 2026-07-27 | ADR-014 |
| 9 | Subsample cohort mismatch: 2.2% row overlap at N=400; selftest 98% NaN | ADR-014 |
| 10 | Embedding producer/consumer path mismatch (see Priority 1) | ADR-015 |
| 11 | Move-classifier session leakage: all 50 sessions on both sides | ADR-016 |
| 12 | Duplication / stale caches / coin-flip triage | ADR-015 |
| 13 | **A verify check that could never fire**: `submission.verify` resolved its CSV to `_staging/` (the zip *build* dir, which never holds one) while `submission.smoke` writes to `_smoke/`. All nine output checks — including `row_ORDER_matches`, the guard for defect #6 — silently never ran, and the report still read green | ADR-018 |

### The pattern worth generalizing

Defects #5, #6 and #13 are one species: **verification that inspects structure but never
exercises behaviour**, with green checks read as evidence. Hunt that shape:

- guards whose condition can never be true
- broad `except: pass` that converts a failure into a plausible default
- silent defaults where a config or artifact is missing (`CLAUDE.md` requires: *fail fast and
  loud on bad config, never silently default*)
- fallback values that could be mistaken for real predictions
- assertions that compare a value to itself, or to something derived from the same expression
  under test

### Specific things to scrutinize

- **`inference_lib.py` train/serve parity.** It is copied verbatim into the zip and is the
  one place training and serving can diverge. Confirm every serve-time feature is computed
  identically at train time **by running both and comparing arrays**, not by reading. Both
  paths import it, so any divergence must come from state, config, or column ordering — find
  which.
- **`_fold_safe_lo_encoding` in `models/gbdt.py`**, rewritten after defect #8. Verify it is
  genuinely fold-safe: perturb one fold's validation labels, assert no other fold's encoding
  moves.
- **Cross-row features are PROHIBITED** — each test sample must be processed independently.
  `features/assemble.py` holds a `CROSS_ROW_FEATURES` frozenset and
  `verify.verify_no_cross_row_features` checks the bundle. Confirm nothing has crept in. A
  violation here is **disqualification**, not a score penalty.
- **Deployment shrinkage** (`unseen_lo.py`, ADR-017): `p' = base + w·(p − base)`, w=0.55,
  base=0.70247, fitted on a 34,446-row unseen-objective holdout. Verify the holdout is
  genuinely objective-disjoint **and** session-disjoint, that `w` is not fitted on rows used
  to train the boosters, and that the applied `base_rate` matches the fitted one.
- **`docs/FINDINGS.md` contains numbers measured under defect #8** — the ablation table and
  the `evaluate.unseen_lo` figures are stale and need re-running. Flag any other doc claim a
  fresh run contradicts. This entry competes for a **publication bonus**, so a wrong number in
  the write-up is a real cost, not a cosmetic one.

---

## Deliverables

Per finding:

1. **File and line**, plus a one-sentence statement of the defect.
2. **A concrete failure scenario** — inputs/state → wrong output. If you can't construct one,
   label it *suspicion*, not *bug*.
3. **Evidence you actually ran**: command and real output. "Looks wrong" is not a finding; a
   leakage claim needs the perturbation experiment, not an argument.
4. **The fix**, plus a regression test that fails before and passes after.
5. **Severity**: does it change predictions, waste units, or risk disqualification?

Then give me:

- an ordered list of what you changed and why
- anything suspicious you could **not** confirm — a ranked list of doubts beats false confidence
- **anything wrong with the approach itself**, not just the code. The CV-to-LB gap is ~0.068
  and calibration turned out to be worth more than every feature combined; say so if you think
  the modelling strategy is misdirected.

Gates that must be green when you finish:

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/pytest
.venv/bin/python -c "
import traceace; traceace.configure(repo_dir='.')
from traceace import tasks; tasks.run('selftest.all')"
```

plus `submission.verify` at **24 checks, 0 failures**.

**Do not rebuild `submission/submission.zip`** unless a fix requires it. A verified artifact
is on disk (md5 `8e505946aa830014c384f8ac97259ad4`) queued for submission. If your changes
invalidate it, say so **loudly, at the top of your report**.
