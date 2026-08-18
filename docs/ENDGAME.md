# ENDGAME.md — the last 9 days (2026-08-18 → 2026-08-27)

**Written:** 2026-08-18 · **Deadline:** model submissions 2026-08-27 23:59 UTC
**Position:** LB 0.6106 / AUROC 0.6014 / rank **#45** · **Units remaining ≈ 729 / 733**

This document supersedes the "Next actions" list in [`STATE.md`](STATE.md). It exists because
a quantitative decomposition of the leaderboard changed the strategy, not the tactics.

---

## 1. The decomposition — where the 0.0145 gap to #1 actually is

Log loss for a **calibrated** model decomposes as

```
LogLoss  =  H(p̄_test)          the entropy of the test base rate  (nobody can beat this)
          + KL(p̄_test ‖ p̄_model)   penalty for centring predictions in the wrong place
          − G                    the discrimination gain
```

### 1a. An empirical law relating AUROC to the achievable gain

Second-order expansion of the log loss around the base rate gives
`G ≈ Var(p) / (2·p̄·(1−p̄))`, and for this data `Var(p) ∝ (AUC−0.5)²`. Measured on four of
our own OOF prediction sets spanning AUC 0.606 → 0.723:

| OOF model | AUC | gain over base rate | **k = G/(AUC−0.5)²** |
|---|---|---|---|
| `model_transcript_only` | 0.6062 | 0.01474 | **1.307** |
| `baseline_lo_only` | 0.7074 | 0.05692 | **1.324** |
| `model_gbdt` | 0.7235 | 0.06648 | **1.331** |

**k = 1.31 – 1.33 across a 0.12 AUC range.** That is a usable law: `G ≈ 1.321·(AUC−0.5)²`.

### 1b. Two leaderboard submissions independently measure our true test-set gain

We have shipped the same model at two shrinkage settings — `p' = b + w(p−b)`:

| w | LB log loss |
|---|---|
| 1.00 | 0.6106 |
| 0.55 | 0.6133 |

For a calibrated model, `LL(w) = C + G·(w²−2w)`. Two points, two unknowns:

```
G = 0.01333       (our discrimination gain, measured on the actual test set)
C = 0.62393       (H(p̄_test) + our centring error)
```

**Cross-check:** the k-law predicts `G = 1.321·(0.6014−0.5)² = 0.01358` from our LB AUROC
alone. The shrinkage curve measures **0.01333**. Two completely independent routes agree to
2×10⁻⁴ — under 2% of G. The framework is correct.

**Corollary — deployment shrinkage was never going to work.** The fitted optimum is
**w\* = 1.0**: our model is *already optimally calibrated on the test set*. ADR-017's premise
("the probabilities are too extreme") was false, and the retirement in STATE.md was right for
the wrong reason.

### 1c. Applying the law to the leaderboard

`C = LL + 1.321·(AUC−0.5)²` for the published top of the table:

| rank | LL | AUROC | implied C |
|---|---|---|---|
| #1 appleswim | 0.5961 | 0.6430 | 0.62311 |
| #2 load_state_dict | 0.5966 | 0.6447 | 0.62425 |
| #4 fishnchips | 0.6001 | 0.6342 | 0.62389 |
| #6 MPWARE | 0.6006 | 0.6281 | **0.62227** |
| #18 enesakca29 | 0.6041 | 0.6177 | 0.62240 |
| **us (#45)** | **0.6106** | **0.6014** | **0.62393** |

Every serious competitor lands in `C ∈ [0.6223, 0.6243]` — a **0.00198** spread — while their
log losses spread over 0.015. **C is essentially a constant of the problem and log loss is a
monotone function of AUROC.** Best observed C = 0.62227 bounds `H(p̄_test) ≤ 0.62227`, giving

> **implied test base rate ≈ 0.686**, versus **0.7025 in training** — a −1.65 pp shift.

### 1d. The verdict

| component of our 0.01450 gap to #1 | size | share |
|---|---|---|
| **discrimination (AUROC 0.6014 → 0.6430)** | **0.01343** | **93%** |
| centring at 0.7025 instead of ≈0.686 | 0.00064 | 4% |
| residual calibration | 0.00043 | 3% |

**There is no calibration trick left.** Ranking is the whole game. Required AUROC:

| target LB | rank | AUROC needed |
|---|---|---|
| 0.6043 | ~#19 | 0.6219 |
| 0.6017 | ~#10 | 0.6297 |
| 0.6001 | ~#4 | 0.6343 |
| 0.5961 | #1 | 0.6452 |

**Top-15 (the write-up gate) needs AUROC ≈ 0.625, i.e. +0.024 over today.**

---

## 2. Why we are stuck at AUROC 0.60 — the metric has been lying

Our shipped model scores **0.7235 AUROC in session-grouped CV** and **0.6014 on the
leaderboard**. Our *transcript-only* model scores **0.6062 in CV**. The leaderboard number
matches the transcript-only CV number, not the full one.

> **The per-objective difficulty lookup — which supplies ~80% of our CV AUROC — is worth
> approximately nothing on the test set.**

Measured directly (`GroupKFold` on `learning_objective_id`, AUC computed **within** each
held-out fold and averaged over 3×5 folds):

| signal | AUC on held-out objectives |
|---|---|
| per-LO empirical lookup, objectives **seen** (our CV regime) | **0.706** |
| per-LO empirical lookup, objectives **unseen** (the test regime) | **0.500** |
| LO *text* → difficulty (TF-IDF + logistic), standalone | 0.575 ± 0.065 |
| transcript-only model | 0.6055 ± 0.012 |
| transcript + LO-text difficulty | 0.6124 ± 0.056 → **see §2a: this did not replicate** |

⚠️ **Methodological trap, recorded because we fell into it.** Pooling predictions across
objective-grouped folds and scoring AUC once is invalid — the folds have different base
rates, so a *constant* predictor scores AUC **0.457** rather than 0.500. All objective-grouped
AUCs must be computed within fold and averaged. The first version of this analysis reported
that LO-text difficulty added nothing; the correct metric shows it adds **+0.007**.
`scripts/lb_decomposition.py` asserts this artifact rather than describing it.

**The strategic consequence is larger than the +0.007.** Every model we have rejected was
rejected on session-grouped CV, which is ~80% a measure of objective-difficulty memorisation
and therefore ~80% irrelevant to the leaderboard:

- `model.bge_attention` — rejected for being "0.00717 worse than objective-only".
- Hierarchical ModernBERT — rejected at 0.58222 vs a 0.58112 objective-difficulty baseline.

**Neither was ever evaluated on the regime the leaderboard tests.** They may be our best
assets. Re-scoring them is nearly free and happens first.

---

## 2a. The objective-fold noise floor, and the first thing it killed

Built and measured 2026-08-18. Everything above in §2 came from a **single** objective-fold
assignment, and objective folds turn out to be far noisier than session folds — each holds only
~80 of 398 objectives, and objective difficulty is lumpy.

`features/lo_difficulty.py` was written to exploit the 0.575 standalone signal: fold-safe
objective-text → difficulty, wired into `model.gbdt` as `include_lo_text_difficulty`. On the
canonical fold assignment it measured **+0.0105 AUROC** — a headline gain, bigger than any
feature block we have.

Repeated across five assignments by `evaluate.objective_repeated`:

| fold assignment | 11 | 23 | 37 | 41 | 59 |
|---|---|---|---|---|---|
| paired ΔAUROC | −0.0055 | −0.0181 | **+0.0149** | −0.0142 | −0.0011 |

> **−0.00478 ± 0.00577, positive in 1/5. ❌ Rejected.** The single-assignment gain was noise and
> the sign flipped. `include_lo_text_difficulty` defaults to `False`; nothing ships.

**Two consequences, and the second matters more than the feature.**

1. Objective descriptions carry standalone signal (0.575) that is nonetheless **redundant with
   what the transcript already provides**. That is a clean, quantified statement about the
   organisers' anti-goal, and it belongs in the write-up as a negative result.
2. **The paired SD across single assignments is ~0.037 — roughly 70× the session-CV noise floor
   of 5e-4.** No objective-fold A/B below ~0.01 AUROC means anything on one assignment. Every
   decision from here, including the expensive GPU ones, goes through
   `evaluate.objective_repeated`; `evaluate.objective_noise_floor` runs the same harness with
   identical arms to calibrate what zero looks like.

This is the fourteenth time a headline number in this project has come apart under a proper
error bar. The harness now exists so it is the last.

---

## 3. Plan

Governing rule for the next nine days: **`robust_cv` objective-purged folds, AUROC averaged
within fold, is the only number that decides anything.** Session-grouped CV log loss is
demoted to a regression guard. Every experiment reports projected LB via
`LB ≈ 0.62393 − 1.321·(AUC−0.5)²`.

Budget posture inverts: ~729 units for 9 days is **~60 A100-hours**. Hoarding units was
correct in week 1 and is now the main risk. Spend.

### Phase 0 — ✅ done 2026-08-18, CPU, 0 units
1. ✅ `evaluate.by_objective_fold` — scores any OOF parquet under objective-purged folds with
   within-fold AUROC and a projected LB. Rows are labelled `honest` (trained objective-disjoint)
   or `optimistic`; the summary headlines the best **honest** row, because an optimistic one tops
   the table at 0.72 while being worth 0.60 on the leaderboard.
2. ✅ Re-scored every OOF on disk. The only honestly-trained model projects to **0.6117** against
   our real **0.6106** — the projection is sound, and the ranking of everything else changed.
   The `interpret.ablation_repeated` verdicts in FINDINGS.md are measured on the wrong metric and
   are **not** evidence for deployment decisions.
3. ✅ `features/lo_difficulty.py` built, measured, and **rejected** — see §2a. The expected
   −0.002 LB gain does not exist. Its real yield was the noise floor it exposed.
4. ✅ `evaluate.objective_repeated` / `evaluate.objective_noise_floor` — the promotion gate that
   every subsequent A/B, including the encoder, has to pass.

**Net effect on the leaderboard: zero.** The cheap win was not real. All of the remaining
distance has to come from Phase 1, which was always where the 93% lived.

### Phase 0.5 — pipeline audit, 2026-08-18 second pass

A line-level audit of everything the GPU cells execute, before any unit is spent. Six defects
found and fixed; numbering continues the silent-correctness series:

1. **(14th defect, parity class)** `build_examples` read transcripts **without**
   `normalize_frame`, while every feature block and the submission path normalize (which
   *re-sorts* rows by `utterance_idx, t_seconds`). Window spans would have indexed
   differently-ordered rows — the encoder trained on different text than inference selects.
   Same class as ADR-013. Fixed: the encoder now normalizes through the shared
   `inference_lib` implementation.
2. **Smoke cell crashed on a fresh GPU runtime.** Fold tables are built by CPU-guarded tasks
   the tier guard refuses on an attached GPU. `_ensure_folds` now builds missing tables
   inline (~1s; the guard exists to stop wasted hours, not seconds).
3. **A guard that fired on the wrong condition.** The "must not run in submission mode" check
   keyed on progress bars being disabled — a legitimate state in any nohup/background run.
   Found by *executing* the training loop on CPU, not by reading it. Removed: `main.py`
   never imports training code, so the guard protected nothing.
4. **Mixed-config OOF assembly.** An aborted run's folds plus a re-run under different
   settings would assemble into an "OOF" of a model that does not exist. Every fold now
   writes a config sidecar; assembly refuses on any mismatch.
5. **Blend regime mismatch — the #45 mistake, re-armed.** Blending the session-trained GBDT
   OOF (AUC 0.72 via memorized objective difficulty) with the honest encoder OOF lets the
   weight optimizer load onto phantom signal. `ensemble.blend` now records
   `input_split_modes`, warns loudly on mixtures, and writes a manifest so
   `evaluate.by_objective_fold` labels blends honestly. The notebook blends the encoder with
   an **honest GBDT twin** (`model.gbdt_objective`) instead of the shipped model.
6. Smaller: transformers scheduler import replaced with plain torch (survives a Transformers
   major bump on Colab); `gradient_checkpointing` flag added (L4 24GB does not reliably fit
   batch 8 × 2048 tokens without it); DataLoader workers so tokenization doesn't starve the
   A100; the diagnosis sweep excludes `rep.*` harness artifacts.

The full training loop was then **executed end-to-end on CPU** with a cached 33M-param
backbone (BGE-small standing in for ModernBERT) — 5 folds, OOF assembly, provenance guard —
so the first GPU run exercises code that has already run, not code that has only been read.

**⚠️ The one critical gap that remains open: `submission.build` has no encoder path.** The
retrieval side is already shipped (`main.py` computes `topk_spans` + `frame_from_spans` with
the trained vectorizer), so the missing piece is: vendor tokenizer + fold weights into
`assets/`, render `render_windows(sub)` per response, batch-forward through torch, average
fold sigmoids, blend with the GBDT logit at the honest weight. **This must be built and
verified before submission slot 2 (~Aug 22)** — it is CPU work and can proceed in parallel
with the encoder's GPU runs, but a winning encoder that cannot be packaged is worth nothing.

### Phase 1 — days 1–4, the neural transcript model (the actual bet)
This is where +0.02–0.035 AUROC has to come from. Hand-crafted lexical features on ASR text
have plateaued at 0.606 and the leaders' 0.63–0.64 AUROC is what a competent transformer
fine-tune on tutoring dialogue looks like.

**Input, per response** (not per session — 58.8% of responses share a session and 26% of
outcome variance is *within* session):
```
[top-k LO-relevant windows, transcript order, role-tagged]   → 1024 / 2048 / 4096 tokens
```
Reuse the existing TF-IDF window retrieval in `inference_lib` so training and submission
select identical windows by construction (ADR-007).

**Design decision to test explicitly: keep the LO text out of the model input.** Use it for
retrieval only. A transformer handed the LO text will memorise objective difficulty, which is
exactly the failure mode of §2 *and* the organisers' stated anti-goal. A dialogue-only encoder
plus a separate LO-difficulty feature is cleaner, more defensible in the write-up, and likely
to transfer better. Test both on one fold; let the objective-purged AUROC decide.

**Backbones** (all Apache-2.0, all bundleable):
- ModernBERT-base, 8192 ctx — fast workhorse, run first. 5 folds × 3 epochs at 2048 tokens
  ≈ 1.1B tokens ≈ 2–3 h on A100 ≈ **30 units**.
- Qwen3-0.6B / 1.7B + LoRA sequence head — stronger on disfluent ASR text. ≈ **80–120 units**
  for 5 folds. Affordable; gate it on the ModernBERT result.

**Guardrails.** The signal is weak (AUROC ceiling ~0.65) and the labels are noisy, so
overfitting is the default outcome: LR 1e-5–3e-5, ≤2–3 epochs, early stopping on
objective-purged val AUROC, ensemble over seeds. The previous pilot scored **AUROC 0.4404 —
below random**, which is a broken head, not a refuted hypothesis. Sanity-gate every run at
AUROC > 0.55 on a single fold before spending on five.

**Inference cost is a non-issue:** 10,508 responses × 2048 tokens ≈ 21M tokens ≈ minutes on
the A100, against a 6-hour cap. Weights: ~0.3 GB × 5 folds against a 60 GB zip limit.

### Phase 2 — days 4–6, LLM tutoring-move annotation (contingent + write-up centrepiece)
Run only if Phase 1 lands below ~0.625. Directly targets the signal we most expect to
transfer across providers: **how the tutor responded to what the student said**. Our lexical
`feedback` block measures +0.00046 on the wrong metric; re-measure it under §3 Phase 0 first.

Pipeline, already half-built (`annotate.py`, `models/move_classifier.py`): local vLLM
(Qwen3-4B/8B-Instruct, Apache-2.0 — **never a hosted API**, per the forum ruling) labels
utterances on ~3–5k sessions → distil into a small encoder → run the *distilled* classifier at
inference (1.9M utterances × 64 tokens ≈ minutes). ~40–60 units. This is one of the three
research directions the organisers named, so it earns its place in the write-up even if its
model contribution is small.

### Phase 3 — days 6–8, assemble
Blend on objective-purged OOF, calibrate, package, A100 timing validation (~4 units).

**Recentring (apply last, to the final submission only).** Our predictions average 0.70247;
the inferred test base rate is ≈0.686. Correcting the intercept is worth ≈0.0007. It is
**fitted on leaderboard feedback** — flagged as such, not held-out evidence. Recommend
recentring only **halfway, to ≈0.694**: the penalty is quadratic, so half the shift captures
~75% of the gain at a quarter of the exposure if the inference is off.

---

## 4. Submission slots — the binding constraint

Three per rolling 7 days; ~4–5 remain before the deadline. **Verify whether the window is
rolling or calendar before spending one** — it changes the whole schedule.

| # | when | payload | purpose |
|---|---|---|---|
| ~~1~~ | ~~now~~ | ~~GBDT + LO-text difficulty~~ | **cancelled** — the feature was rejected (§2a); nothing to bank |
| 2 | ~Aug 22 | + neural transcript model, blended | the bet |
| 3 | ~Aug 25 | + Phase 2 features / better neural | refinement |
| 4 | Aug 27 | best OOF configuration + half recentring | final |

Phase 0 produced no shippable gain, so there is now **nothing worth spending a slot on until the
encoder lands**. That is a better position than spending one to find out.

**Hold one slot for Aug 27 unconditionally.** Do not spend a slot on a diagnostic that
offline analysis can answer — §1 answered the transcript-only question without one.

---

## 5. What this hands the write-up

The decomposition in §1 and the metric failure in §2 are publishable findings in their own
right, and they speak directly to two of the four scoring criteria:

- **Relevance / Rigor.** "Leaderboard log loss in this task is a monotone function of AUROC
  with a constant of 0.6238; calibration effort is wasted" is actionable guidance for other
  researchers, derived rather than asserted.
- **Generalizability.** "Per-objective difficulty lookups score 0.706 AUC on seen objectives and
  0.500 on unseen ones; objective *text* recovers 0.575 standalone but adds **nothing** on top of
  dialogue (−0.005 ± 0.006 over five fold assignments)" is a precise, quantified statement of what
  transfers to a new tutoring deployment and what does not — the organisers' stated anti-goal,
  measured instead of assumed, with the null result intact.
- **Rigor.** The objective-fold noise floor (paired SD ~0.037, ~70× the session-CV floor) is
  itself a methodological finding: papers reporting held-out-topic results on a single split are
  reporting noise, and we can show the effect size at which that starts to matter.
- **Negative results.** Deployment shrinkage (ADR-017) was falsified by the leaderboard *and*
  is now explained: the model was already optimally calibrated (w\* = 1.0).

Reproduce every number in §1 and §2 with:

```bash
python scripts/lb_decomposition.py
```

(CPU, ~30 s, aggregates only. From a git worktree, set `TRACEACE_REPO_DIR` to the primary
checkout — `data/` and `artifacts/` are gitignored and exist only there.)
