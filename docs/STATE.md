# STATE.md — read this first

**Last updated:** 2026-08-19 · **CI: green ✅**

> ## ⚡ 2026-08-19 — encoder campaign status (supersedes everything below; details in [`ENDGAME.md`](ENDGAME.md))
>
> **Fold-0 probes on the A100 (67 units spent, ~666 left):**
>
> | config | solo AUC | blend w/ honest GBDT (0.6091) | verdict |
> |---|---|---|---|
> | ModernBERT 2048 tok / 4 win | 0.5738 | 0.6167 (w=0.15) | context-starved |
> | **ModernBERT 3072 tok / 6 win** | **0.5979** | **0.6235 (w=0.25)** | ✅ **winner — clears the 0.622 top-15 line on fold 0** |
> | Qwen3-0.6B 2048 / 4 | 0.5774 | 0.6173 (w=0.20) | 3× cost, no gain — rejected |
>
> Lesson: the bottleneck was dialogue *quantity*, not model capacity. Epoch curves:
> wide config still climbing at epoch 3 (0.534→0.593→0.598); baseline overfits after ep2.
>
> **Full 5-fold run** (`experiment="probe.enc_wide"`, epochs=3, lr 3e-5, 3072/6,
> objective-disjoint, ~66 units): operator launches/launched it with
> `shutdown_after=True`. It auto-assembles the OOF and prints the honest
> `within_objective_fold_auc` + projected LB. Fold 0 was the GBDT's best fold, so the
> 5-fold mean will read below 0.6235 — that is composition, not decay.
>
> **Encoder packaging is BUILT and VERIFIED** (`f480c51`): weights/tokenizer/config
> vendored, offline inference in `encoder_lib.py`, logit blend at an explicit
> `encoder_weight`, 28/28 verify checks green incl. parity + OOF replay through the
> blended pipeline. Smoke: 100 rows in 64 s → projected **1.87 h** vs the 6 h cap.
>
> **Next, in order:** (1) read the full-run OOF number; (2) blend with
> `model.gbdt_objective`, re-run `evaluate.by_objective_fold`; (3) if the blend holds
> ≥ ~0.615, `submission.build(experiment="model.gbdt", encoder_experiment="probe.enc_wide",
> encoder_weight=<measured>)` → smoke → **spend a slot**; (4) levers left: second-seed
> encoder ensemble (~+0.003–0.005), move-taxonomy features (write-up centrepiece).
> Recentring half-way to ≈0.694 is applied only to the FINAL Aug-27 submission.

All numbers below are mean ± SD over 5 fold assignments (2026-07-31 state).
**Competition deadline:** model submissions 2026-08-27 23:59 UTC · write-up 2026-09-15
**Entry:** solo · **Units remaining: 728.97 / 733**

> New session? Read [`../CLAUDE.md`](../CLAUDE.md) → [`BRIEF.md`](BRIEF.md) → this file.
> Then [`DATA.md`](DATA.md) for measured facts and [`RUNBOOK.md`](RUNBOOK.md) for recipes.

---

## Status: SUBMISSION 2 SCORED 0.6106 · rank #45 · one slot remains this week

**Leaderboard history**

| # | id | log loss | AUROC | rank | verdict |
|---|---|---|---|---|---|
| 1 | 2723 | 0.8006 | 0.4933 | #229 | 🔴 **broken** — feature-order permutation (ADR-013) |
| 2 | — | — | — | — | (slot spent on a smoke test) |
| 2 | — | **0.6106** | **0.6014** | **#45** | 🟡 works, but **overconfident** |

**Submission 3 is the real baseline.**

> ⛔ **2026-08-18 — the diagnosis below was WRONG. Read [`ENDGAME.md`](ENDGAME.md) first.**
> The two shipped shrinkage settings (w=1.00 → 0.6106, w=0.55 → 0.6133) fit
> `LL(w) = C + G(w²−2w)` exactly, giving **w\* = 1.0**: the model was *already optimally
> calibrated on the test set*. It was never overconfident. The fitted discrimination gain
> G = 0.01333 matches the independent prediction from our LB AUROC (0.01369) to 4×10⁻⁵.
> **87% of the gap to #1 is AUROC, not calibration**, and the per-objective difficulty lookup
> — ~80% of our CV AUROC — is worth ≈0 on unseen objectives. Session-grouped CV has been
> selecting models on a metric that is ~80% irrelevant to the leaderboard.
> The remainder of this section is retained as the record of a superseded hypothesis.

~~AUROC 0.6014 says the ranking carries genuine signal;
0.6106 log loss says the probabilities are too extreme for how hard the test regime is. That
gap is *not* brokenness and not noise — it is structural, and it is bigger than every feature
gain we have measured combined.~~

**~~Diagnosis.~~** Our CV regime is easier than the test regime: 99.7% of validation rows have an
objective seen in training, and every session comes from one provider. The test set contains
unseen objectives and (per the forum) more than one provider. ~~A model tuned to be confident on
the easy regime is systematically overconfident on the hard one.~~ The domain shift is real; its
consequence is **lost discrimination**, not miscalibration.

**Fix, ready to ship: deployment shrinkage** (ADR-017). `p' = base + w·(p − base)`, with
**w = 0.55** and base = 0.70247, fitted on a 34,446-row unseen-objective holdout — never on
leaderboard feedback. Ranking-invariant by construction (AUC identical to 6 decimals).

| on the unseen-objective holdout | log loss |
|---|---|
| unshrunk (w = 1.0) | 0.59727 |
| shrunk, evaluated on the same rows used to fit w | 0.59012 |
| **shrunk, response-disjoint cross-fit** | **0.59181** |
| constant prior | 0.59849 |

The original **+0.00715** estimate was in-sample calibration performance. Round 3 corrected
the evaluation: response-disjoint cross-fitting gives **+0.00546**. The direction remains
strong and materially larger than any individual feature block, but the earlier estimate was
optimistic and must not be quoted as held-out evidence.

❌ **Deployment shrinkage is retired.** Its leaderboard run worsened log loss from 0.6106
to 0.6133 at identical AUROC. Packaging defaults to no shrinkage and requires an explicit
research-only opt-in to reproduce it.

🔍 **ADR-018 — a verify check that could never fire.** Found while re-verifying the above.
`submission.verify` looked for its CSV in `_staging/` (the zip *build* dir, which never holds
one) while `submission.smoke` writes to `_smoke/`. **All nine output checks — including
`row_ORDER_matches`, the guard for the exact bug that broke submission #1 — silently never
ran** in any standalone verify, and the report still looked green. Fixed; ordering is now
checked exactly, against the format each run was actually handed. A new
`all_expected_checks_ran` check asserts every guard by name, so a check going dark is now
itself a failure. **Thirteenth silent-correctness defect; third of the species
"verification inspects structure, never behaviour."**

⚠️ **Current operator action:** build and verify the freshly retrained transcript-only
candidate. Do not upload until its manifest, smoke execution and complete verifier report
have been inspected.

⚠️ **Stale numbers to redo:** `interpret.ablation_repeated` and `evaluate.unseen_lo` were both
measured *before* the target-encoding leakage fix (ADR-014). Their FINDINGS.md figures are
superseded and must be re-run before the write-up quotes them.

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
**Measured LB (submission 3): 0.6106 / AUROC 0.6014.** The CV-to-LB gap is ~0.068, far
larger than the seen/unseen spread above — which is what motivated ADR-017.

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
- **All 37 tasks** registered and runnable via `traceace.tasks.run(name)`.
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
- **Submission**: 1.63 MB zip, `main.py` at root, **all 26 verify checks pass** — including
  feature-order, the nine output checks (now actually reachable, ADR-018), prediction-sanity
  on training data, the coin-flip line, and `all_expected_checks_ran`. Smoke ~4 s →
  **~0.13 h projected** for the full test set (cap 6 h).
  `prediction_sanity` gates on **AUC 0.8285** (primary) with a deliberately loose log-loss
  bound.
  Value-level parity now compares every deployed feature against the training caches
  (**0 mismatched cells across 180 columns**), and packaged fold models replay **626/626**
  held-out OOF predictions exactly, including fold-specific objective priors.
- **Quality gates**: ruff clean · mypy clean (49 files) · **94 tests pass** ·
  `selftest.all` green in ~25 s · **GitHub Actions CI green**.
  ⚠️ **Treat a green suite as weak evidence.** These same gates were green while **thirteen**
  correctness defects were live, four of which reached a submission, one of which scored
  below random on the leaderboard, and one of which was a *verification check that never
  executed*. See ADR-013 … ADR-018.
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

## Smoke-score triage — the coin-flip line
Constant-0.5 predictions score **ln(2) = 0.6931** on any binary labels. Above that line a model
is *confidently wrong* (scrambled features, misaligned rows), not merely weak. Smoke history:
`id-2719` 0.8543 ❌ · `id-2721` 0.8330 ❌ · **`id-2728` 0.4686 ✅** (matches the local
training-data probe, 0.44065). Enforced by `verify.beats_coin_flip`.

## Move-classifier: honest baseline
Full-scale, session-disjoint evaluation on 45,642 heuristic annotations:
**0.8739 accuracy / 0.7722 macro-F1**. Supersedes both the leaky 0.7650 and the
smoke-cohort 0.3250 (ADR-016). Caveat: heuristic labels are regex-derived, so the classifier
is largely reproducing rules — this does **not** establish it can reproduce LLM-quality labels
at that level, which is what the ~40-unit `annotate.moves` plan actually depends on.

## Known problems / open risks
1. **The margin is thin but now solid** (−0.01132 ± 0.00066, CI excludes zero on 5/5 seeds).
   Most of the model's power is still the topic prior.
2. **Only 2 of 6 blocks are distinguishable from zero.** Redundancy is contextual: LO-alignment
   went from +0.00059 to −0.00030 when trajectory was added. Re-run the ablation after EVERY
   addition; never trust a stale one.
3. **`main.py` was missing 40% of its features until 2026-07-26** — the feedback, trajectory
   and LO-position blocks were never wired in, arriving as NaN. Fixed, and now impossible to
   repeat: `verify_feature_coverage` fails the build on any gap (ADR-010).
4. **Semantic BGE extraction is complete but not promoted.** Full L4 extraction encoded
   601,459 windows for 22,821 sessions in 48 minutes (4.02 units). Honest paired evaluation:
   content improves `+0.00039 ± 0.00032` (5/5, but below the 0.001 deployment gate), while
   alignment improves `+0.00016 ± 0.00049` (3/5, interval includes zero). Neither enters the
   submission.
5. **`annotate.moves` has only run with the `heuristic` backend.** The vLLM backend is
   written but unexercised.
6. Local env now has **torch (CPU) + sentence-transformers** installed so GPU code paths can
   be validated before spending units — a deliberate relaxation of ADR-005.
7. Git SHAs in early run manifests read `unknown` (they predate the first commit).
8. **Hierarchical ModernBERT is rejected.** The 3,000-session A100 pilot scored 0.58222
   versus a 0.58112 training-rate baseline (AUROC 0.4404); its conditional head was
   actively harmful. Five-fold/full-data training is disabled in the runner.
9. **Transcript-only transfer now has genuine hard-fold evidence.** An 80-round diagnostic
   retrained on domain-cluster holdouts scored 0.59494 / AUC 0.6052 and beat the fold-local
   prior in 5/5 folds. Purged objective holdouts scored 0.59706 / AUC 0.5958 and also beat
   the prior 5/5. These are deliberately under-tuned diagnostics, not submission estimates.

## Leaderboard evidence (supersedes shrinkage simulations)
- Unshrunk deployable model: **log loss 0.6106 / AUROC 0.6014 / rank 45**.
- Shrinkage `w=0.55`: **log loss 0.6133 / AUROC 0.6014 / rank 54**. Identical AUROC and
  worse log loss falsify the deployment-shrinkage hypothesis; retire this candidate.
- Full grouped CV AUROC (~0.723) materially overstates transfer. Transcript-only grouped CV
  AUROC is **0.6062**, close to leaderboard AUROC, so objective difficulty is the main
  validation-to-test domain-shift risk.
- A deployable transcript-only fallback now trains via
  `model.gbdt(include_lo_prior=False)`. It is a robustness candidate, not yet a claimed win.
- `model.bge_attention` completed: **0.54467 log loss / 0.7205 AUROC**, but it was **0.00717
  worse than objective-only**. The familiar high session-CV AUROC shows that semantic queries
  reconstructed objective difficulty; this route is rejected for deployment.

## Next actions, in order

> **Superseded 2026-08-18 by [`ENDGAME.md`](ENDGAME.md)** — that document is the operative
> plan for the remaining 9 days. Summary of the change: the objective-purged fold AUROC,
> averaged **within** fold, replaces session-grouped CV log loss as the metric that decides
> anything; unit hoarding ends (~729 units for 9 days ≈ 60 A100-hours); the priority is a
> fine-tuned transcript encoder, because 87% of the leaderboard gap is discrimination.
>
> A transcript-only diagnostic submission — previously step 1 here — is **no longer worth a
> slot**: §1 of ENDGAME.md answers it offline.

1. ✅ `evaluate.by_objective_fold` — within-fold AUROC harness. Everything depends on it.
   Best *honest* model projects to 0.6117 vs our real 0.6106; the projection is sound.
2. ✅ `evaluate.objective_repeated` + `evaluate.objective_noise_floor` — the promotion gate.
   **The objective-fold paired SD is ~0.037, ~70× the session-CV noise floor.**
3. ❌ `features/lo_difficulty.py` — built, measured, **rejected**: −0.00478 ± 0.00577 AUROC over
   5 fold assignments, positive in 1/5. The +0.0105 single-assignment reading was noise.
   `include_lo_text_difficulty` defaults False; nothing ships. See ENDGAME.md §2a.
4. ⬅️ **NEXT: `model.transcript_encoder`** — response-level, retrieval-selected windows, dialogue
   only (ENDGAME.md §3 Phase 1). L4 smoke → A100 fold 0 (gate: AUROC ≥ 0.60) → 5 folds.
   All of the remaining distance has to come from here.
5. Move taxonomy via local vLLM — contingent on step 4, and the write-up centrepiece regardless.
6. **A100 timing validation (~4 units)** only immediately before a real submission.
7. Tuning the existing tree is **not** a next action — a capacity sweep showed the plateau.
8. **Always** re-run ablations after adding a deployable block; substitutability
   means every prior ablation is stale. All current ablation verdicts are additionally stale
   because they were measured on session-grouped CV.

## Compute budget
733 units, **8.33 spent** at the last A100 pilot. Plan (ADR-008): ~0 CPU ladder · ~5 embeddings · ~40 LLM
annotation · ~30 classifier iteration · ~15 A100 timing. **~600 held in reserve until
week 3.** Flag anything projected above 25 units for a single task.

## Where the numbers came from
`runs/` holds a manifest per task run (git SHA, seed, wall time, units, metrics) and
`runs/budget.jsonl` is the unit ledger. [`EXPERIMENTS.md`](EXPERIMENTS.md) is regenerated
from them by `docs.build` — never edit it by hand.
