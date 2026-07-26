# STATE.md — read this first

**Last updated:** 2026-07-26 · **CI: green ✅** · all numbers are mean ± SD over 5 fold assignments
**Competition deadline:** model submissions 2026-08-27 23:59 UTC · write-up 2026-09-15
**Entry:** solo · **Units remaining: 733.00 / 733** (everything so far ran on CPU, 0 units)

> New session? Read [`../CLAUDE.md`](../CLAUDE.md) → [`BRIEF.md`](BRIEF.md) → this file.
> Then [`DATA.md`](DATA.md) for measured facts and [`RUNBOOK.md`](RUNBOOK.md) for recipes.

---

## Status: SUBMISSION READY (safe variant). Awaiting operator to submit.

✅ **Container smoke tests: 2 run, both passed.**
- `id-2719` — exit 0, ~5 s. Surfaced a scikit-learn version mismatch (ours 1.9.0 vs container
  1.8.0) warning of "invalid results" *while still exiting 0*. Fixed by pinning + shipping the
  calibrator as plain numbers (ADR-012).
- `id-2721` (sha `7d097ff`) — exit 0, ~5 s, **no warnings of any kind**. Fix confirmed in the
  real container. Six clean status lines, `data/` mounted as expected.

Smoke scores (0.8543, 0.8330) are on **fake data** and carry no signal — our model predicts
~0.70 against roughly balanced random labels, which lands right about there.

🔴 **SUBMISSION 1 (id-2723) SCORED 0.8006 / AUROC 0.4933 — rank #229. It was BROKEN, not
merely weak.** A feature-ORDER bug shipped a scrambled design matrix: the bundle took its
column list from `importance.parquet` (sorted by gain), which permuted 179 of 181 positions
against the training order, and LightGBM reads positionally. Every structural check passed.
Fixed in ADR-013; verify now runs the packaged `main.py` against training data and requires
log loss ≤ 0.50 / AUC ≥ 0.75.

Local proof of the fix (packaged `main.py` on training data):
`0.80867 / AUC 0.4738` → **`0.43927` / AUC 0.8291**, mean prediction 0.49 → 0.735 against a
0.7025 base rate.

⏳ **Submission 2 pending** with the fixed artifact. **2 slots left this week.**

✅ **Cross-row rules question: SETTLED 2026-07-27.** Already answered on the forum — cross-row
features are prohibited, our default was correct, nothing to post. +0.00251 permanently
forfeit. See [`FORUM_QUESTION.md`](FORUM_QUESTION.md).

ℹ️ **AI-assistant licensing question: considered and closed 2026-07-27** — treated as
development tooling, not a model contributing content to the solution. Reasoning recorded in
[`OPEN_QUESTION_ai_assistant.md`](OPEN_QUESTION_ai_assistant.md); not posted. The data-handling
practice tightened at the same time stands regardless: no verbatim transcript text in terminal
output or assistant context, aggregates only (see `CLAUDE.md`).

⚠️ **Two things need YOU, not me:**
1. **Submit** — see [`RUNBOOK.md`](RUNBOOK.md) "Submitting for real". I cannot submit on your
   behalf (your account, and it consumes one of three weekly slots). Nothing now blocks this.
2. **The L4 run** — this machine has no GPU. Cells are staged in the notebook.

The full cheap ladder is built, runs on the real data, and produces a valid `submission.zip`.
Everything so far has run on **CPU for zero units**.

## Best score

| Model | CV log loss | AUC | Δ vs `lo_only` |
|---|---|---|---|
| `baseline.prior` | 0.60876 | 0.500 | +0.0569 |
| `baseline.lo_only` (the bar) | 0.55220 ± 0.00022 | 0.707 | — |
| **`model.gbdt` SAFE (shipped, leak-free)** | **0.54286 ± 0.00044** | **0.72263 ± 0.00070** | **−0.00934 ± 0.00040 ✅** |

⚠️ **All CV numbers reported before 2026-07-27 were measured under target-encoding leakage**
(ADR-014) and are superseded. The corrected contribution is smaller: −0.00934 vs −0.01132.
| same model, **unseen objectives only** | 0.59178 ± 0.01143 | — | −0.00804 ± 0.00260 |

The test set contains objectives absent from training, so the true LB score sits **between**
the seen (0.543) and unseen (0.592) figures — at a mix only the organizers know.

**Cost of rules safety: +0.00251 log loss** — we exclude 4 cross-row features pending a forum
ruling (ADR-009). A DQ would end the competition; 0.0025 log loss would not.

5-fold session-grouped CV on all 35,072 responses, **repeated over 5 fold assignments**.
The improvement's 95% CI is [−0.01191, −0.01074] and excludes zero on all 5 seeds.
Full table: [`EXPERIMENTS.md`](EXPERIMENTS.md).
**No leaderboard submission has been made** — LB score unknown.

### Block contributions (paired, 5 seeds) — the only valid basis for block decisions
| Block | Δ (mean ± SD) | Verdict |
|---|---|---|
| trajectory | +0.00226 ± 0.00011 | ✅ real (strongest) |
| linguistic | +0.00174 ± 0.00025 | ✅ real |
| lo_alignment | −0.00030 ± 0.00024 | ⚠️ hurts — features dropped, module retained |
| feedback | −0.00007 ± 0.00014 | ✗ unmeasurable |
| structural | +0.00006 ± 0.00030 | ✗ unmeasurable |
| temporal | +0.00005 ± 0.00047 | ✗ unmeasurable |

**Noise floor: paired SD ≈5e-4; headline varies 0.00105 across seeds.** Never decide a block
on a single-seed difference below ~1e-3.

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
- **Quality gates**: ruff clean · mypy clean (41 files) · **45 tests pass** · `selftest.all`
  green in 23 s · **GitHub Actions CI green** (verified tests actually collect — they
  previously did not).
- **Artifact namespacing**: subsampled runs write to `<experiment>__subN` for OOF *and*
  model dirs, so a smoke run can never corrupt a full-data result or a submission.

## Forum rulings absorbed 2026-07-27 (see COMPETITION.md for the full table)
- 🚫 **Competition data may NOT be uploaded to an API** — `annotate.moves` must use local vLLM.
- ⚠️ **Test set is NOT all Third Space Learning** — our ASR-specific features may not transfer.
  Largest known generalizability risk.
- ⚠️ **Not every test objective appears in training** — measured by `evaluate.unseen_lo`:
  model 0.59178 ± 0.01143 vs prior 0.59982 ± 0.01224; **transcript gain −0.00804 ± 0.00260
  survives**.
- ✅ CC-BY-SA external data allowed; minor non-determinism allowed; best submission
  auto-selected; external-corpus generalizability work explicitly encouraged.

## Known problems / open risks
1. **The margin is thin but now solid** (−0.01132 ± 0.00066, CI excludes zero on 5/5 seeds).
   Most of the model's power is still the topic prior.
2. **Only 2 of 6 blocks are distinguishable from zero.** Redundancy is contextual: LO-alignment
   went from +0.00059 to −0.00030 when trajectory was added. Re-run the ablation after EVERY
   addition; never trust a stale one.
3. **`main.py` was missing 40% of its features until 2026-07-26** — the feedback, trajectory
   and LO-position blocks were never wired in, arriving as NaN. Fixed, and now impossible to
   repeat: `verify_feature_coverage` fails the build on any gap (ADR-010).
4. **Semantic LO-alignment + content block are built but not run at scale.**
   `features.window_embeddings` is validated end-to-end on CPU (40 sessions); full extraction
   needs an L4, projected **35–52 min ≈ 3–4 units**. `features.content` (pooled top-k window
   vectors, PCA-48) is validated and waiting on those vectors. Highest-value pending work.
5. **`annotate.moves` has only run with the `heuristic` backend.** The vLLM backend is
   written but unexercised.
6. Local env now has **torch (CPU) + sentence-transformers** installed so GPU code paths can
   be validated before spending units — a deliberate relaxation of ADR-005.
7. Git SHAs in early run manifests read `unknown` (they predate the first commit).

## Next actions, in order
1. **L4 session — semantic alignment + content block (~3–4 units).** THE highest-value
   experiment. Smoke `run("features.window_embeddings", subsample=500)`, then the full run,
   then `run("features.lo_alignment", backend="embedding", force=True)`,
   `run("features.content")`, and finally `run("interpret.ablation_repeated")` to re-rank.
   Lexical matching is the current ceiling and it now measurably *hurts*; semantic matching
   is the fix, and the pooled content vectors add *what was said* as orthogonal evidence.
2. **Move taxonomy**: `annotate.moves` (vLLM backend) seeded by the feedback categories, then
   `model.move_classifier`, then move-distribution features.
3. **First real submission** once step 1 lands. Budget: 3/week, ~15 attempts left.
4. **A100 timing validation (~4 units)** only immediately before a real submission.
5. Tuning is **not** a next action — a capacity sweep showed the plateau (FINDINGS N6).
6. **Always** re-run `interpret.ablation_repeated` after adding a block; substitutability
   means every prior ablation is stale.

## Compute budget
733 units, **0.00 spent**. Plan (ADR-008): ~0 CPU ladder · ~5 embeddings · ~40 LLM
annotation · ~30 classifier iteration · ~15 A100 timing. **~600 held in reserve until
week 3.** Flag anything projected above 25 units for a single task.

## Where the numbers came from
`runs/` holds a manifest per task run (git SHA, seed, wall time, units, metrics) and
`runs/budget.jsonl` is the unit ledger. [`EXPERIMENTS.md`](EXPERIMENTS.md) is regenerated
from them by `docs.build` — never edit it by hand.
