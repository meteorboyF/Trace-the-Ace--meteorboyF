# DECISIONS.md — architecture decision record

Append-only. Newest at the bottom. Each entry: context, decision, alternatives rejected,
consequences. Dates are absolute.

---

## ADR-001 — 2026-07-25 — Session-grouped cross-validation, always

**Context.** 35,072 responses come from 22,821 sessions; 58.8% of responses sit in
sessions that produce more than one response, and those responses share a byte-identical
transcript.

**Decision.** All folds use `StratifiedGroupKFold(groups=session_id, y=correct)`, generated
once by `cv.build` and persisted to `data/interim/folds.parquet`. Every model reads the
same folds. `tests/test_cv_leakage.py` asserts zero session overlap and runs in CI.

**Alternatives rejected.** Plain `StratifiedKFold` on `response_id` — would place the same
transcript in train and validation, inflating every score into fiction.

**Consequences.** All experiments are mutually comparable and OOF predictions are
blendable. Fold sizes are slightly uneven because grouping constrains the split.

---

## ADR-002 — 2026-07-25 — Architecture follows measured token counts, not the 6h cap

**Context.** The brief anticipated 10–15K tokens/session and a possible inference-time
crisis. Measurement (`eda.transcripts`) shows median **5,323** tokens, p99 8,095, max
11,548 — total corpus ~120M tokens. `eda.inference_budget` projects a chunked encoder at
**0.07–0.10 h** and even a 14B generative model at 2.0–3.1 h, all inside the 6 h cap.

**Decision.** The inference cap is **not** the binding constraint. Architecture is chosen
on signal quality and *development* unit budget instead. Default workhorse is an
encoder-class model with an 8192-token window, which covers ~99% of sessions in one pass.

**Alternatives rejected.** Designing around aggressive truncation or a hierarchical
summarizer to "fit the budget" — unnecessary, and would have discarded signal.

**Consequences.** We can afford richer per-session processing than planned. The
`eda.inference_budget` task stays in the pipeline so this verdict is re-checked if the
test-set shape ever differs from training.

---

## ADR-003 — 2026-07-25 — LO-conditioned features are mandatory, not optional

**Context.** Variance decomposition on the real labels: among multi-response sessions,
38.3% carry *mixed* labels, and **43.3% of total label variance is within-session**. Any
session-level feature — structural, linguistic, temporal, or a whole-session embedding —
assigns identical values to every response in a session and therefore cannot separate
them even in principle.

**Decision.** Every model must consume at least one response-level, LO-conditioned feature
block. `features/lo_alignment.py` provides it: sliding windows over the session scored for
relevance to the learning-objective text, with features pooled over the top-k windows —
including dialogue statistics computed *inside* those windows, which genuinely differ
between two objectives in the same session.

**Alternatives rejected.** (a) Session-level embeddings alone — provably cannot address
43% of the variance. (b) Treating the LO as just another categorical — that is precisely
the organizers' stated anti-goal.

**Consequences.** "Key moments" becomes a modeling requirement rather than a research
side-quest, and the same artifact answers a named research question. Adds a response-level
join to the feature assembly path.

---

## ADR-004 — 2026-07-25 — No generative model in the inference path (dev-time annotation instead)

> **CONFIRMED 2026-07-27 by two organizer rulings.** (a) Bundling open weights for offline
> inference is *"exactly how you should bring in and use open-weight models"*. (b) Closed/hosted
> models are prohibited **even at development time** — *"the open license limitation applies both
> to final models and to development"* — independently of the separate prohibition on uploading
> competition data to an API. Our choice of local vLLM + Apache-2.0 Qwen2.5 was made on cost and
> risk grounds; it is now the only permissible path. A hosted-API annotation backend must never
> be added to `annotate.moves`.


**Context.** A tutoring-move taxonomy is one of three named research directions, and LLM
labelling is the natural way to build one. But putting a generative model in the
submission means vendoring multi-GB weights, a vLLM dependency, and far more runtime risk
against a 6-hour cap and a 60 GB zip limit — for a component whose value is
*interpretive*, not predictive.

**Decision.** Use a generative LLM **dev-time only**, via `annotate.moves`, to label a
stratified sample of *training* utterances. Distil those labels into a small classifier
(`models/move_classifier.py`) that runs over all transcripts at encoder-class cost. The
submission never invokes a generative model.

**Alternatives rejected.** (a) Generative model at inference — cost and risk without
commensurate predictive gain. (b) Rules-only taxonomy — cheap but weaker; retained as the
default `heuristic` backend and as the fallback when the LLM output is unparseable.

**Consequences.** Rules-compliant: annotating training data is permitted, and test samples
remain independently processed with parameters unaffected by test data. Generative
fine-tuning is deferred until the cheap ladder plateaus. Budgeted ~40 units for annotation
and ~30 for classifier iteration.

---

## ADR-005 — 2026-07-25 — Local environment stays CPU-only until an encoder enters the submission path

**Context.** The local machine had Python 3.14 and no ML packages; the competition runtime
is Python 3.12. The entire cheap ladder (baselines, all four feature blocks, LightGBM,
calibration, blending, interpretation, packaging, verification) runs on CPU.

**Decision.** Local dev uses a **uv-managed Python 3.12.13 venv** matching the runtime,
with a **CPU-only** dependency set (no torch/transformers). `selftest.all` — the full
end-to-end pipeline on real data — must pass here. GPU work (embeddings, any encoder
fine-tune) happens only on Colab.

**Alternatives rejected.** (a) System Python 3.14 — diverges from the runtime; would
surface parity bugs late. (b) Installing torch locally — a large download that buys
nothing until an encoder is actually in the submission path.

**Consequences.** Fast, reproducible local verification with a genuinely runtime-matched
interpreter. When an encoder enters the submission path, this ADR is superseded and torch
gets added locally for parity testing.

---

## ADR-006 — 2026-07-25 — Robust statistics for all ASR-derived timing features

**Context.** The corpus is voice-transcribed, not typed chat. `background` (4.0% of
utterances) is a diarization-failure bucket containing real, misattributed tutor speech,
and `[unclear]` appears in ~30% of all utterances. Automatic segmentation produces
spurious long gaps.

**Decision.** All timing features use median / trimmed mean / IQR (`common.robust_stats`),
never mean/std. Gaps above 120 s are treated as breaks and excluded. `background` is kept
(it carries real content) but tracked as its own role, and its volume is exposed as a
data-quality feature.

**Alternatives rejected.** (a) Mean/std — dominated by segmentation artifacts. (b)
Dropping `background` — discards genuine pedagogical speech. (c) Assuming a binary
tutor/student role split — contradicted by the data.

**Consequences.** Features are stable against transcription noise. The ASR nature of the
corpus is a first-class generalizability caveat for the write-up: transfer to *typed* chat
tutoring is non-trivial and must be stated as a limitation.

---

## ADR-007 — 2026-07-25 — One shared feature implementation for training and inference

**Context.** The classic fatal bug in a code-execution competition is train/serve skew:
inference re-implements feature extraction slightly differently, the model receives
subtly different inputs, and the score collapses with no error raised anywhere.

**Decision.** `packaging/inference_lib.py` is a standalone, dependency-light module copied
**verbatim** into `submission.zip`. `tests/test_inference_parity.py` asserts it produces
values identical to the training-time blocks on the same input, key-for-key.

**Alternatives rejected.** Generating inference code by string templating — the first
version of this build did exactly that and was already becoming unreadable and
drift-prone.

**Consequences.** One implementation to maintain. `inference_lib` may not import
`traceace` and may not print (both enforced by tests).

---

## ADR-008 — 2026-07-25 — Compute budget allocation and reserve

**Context.** 733 units, ~5 weeks. Units burn while the runtime is *connected*, so idle
attached GPUs are the largest waste vector.

**Decision.** Planned allocation: ~0 units for the CPU ladder (baselines, all feature
blocks, GBDT, calibration, blending, interpretation, packaging), ~5 for frozen embedding
extraction (L4, once), ~40 for LLM move annotation, ~30 for move-classifier iteration, ~15
for A100 timing validation. **~600 units held in reserve, untouched until week 3.** Any
single task projected above **25 units** must be flagged before running. Rates live in
`conf/base.yaml`, never hardcoded, and are verified against Colab's live Resources panel.

**Alternatives rejected.** Spending early on encoder fine-tuning — the ladder has not been
exhausted, and the measured baselines are strong.

**Consequences.** The tier guard (`max_tier`) enforces this structurally by refusing CPU
tasks on paid runtimes. `budget.report` tracks spend against the 733 balance.


---

## ADR-009 — 2026-07-26 — Ship without cross-row feature inputs (RULING RECEIVED 2026-07-27)

> **UPDATE 2026-07-27 — the question is settled and our default was correct.** The organizers
> had already answered this exact question on 2026-07-09: *"To make a prediction on any given
> test sample, the only input to your model drawn from the test should be that sample's
> metadata and transcript."* Cross-row features are **prohibited**. The decision below stands
> unchanged and is now a rules requirement rather than a precaution; the +0.00251 log loss is
> permanently forfeit. No forum post was needed — searching the existing threads first saved
> both a duplicate post and several days of waiting.


**Context.** Four `lopos_*` features are computed by grouping rows of `test_features.csv`
that share a `session_id`. `lopos_n_competing_los` is the **second-highest-gain feature in
the model**. The rules preclude "using information gathered across multiple test samples as
feature inputs", and the ambiguity is real: `session_id` is provided metadata and no fitting
occurs on test data, but the feature's *value* depends on which other test rows exist.

**Decision.** Default to **excluding** them (`features.allow_cross_row_aggregates: false`),
behind a config flag so the decision is reversible in one line. A forum question is drafted
(`docs/FORUM_QUESTION.md`). `submission.verify` fails the build if a cross-row feature
reaches the shipped model while the flag is off.

**Alternatives rejected.** (a) Ship with them and hope — the penalty is disqualification, not
a worse score, and it would apply to *every* submission made under that assumption.
(b) Delete the features entirely — discards a genuine research finding and cannot be undone
cheaply if the ruling is favourable.

**Consequences.** Measured cost **+0.00251 CV log loss** (0.54088 → 0.54339), about 22% of
the transcript's total contribution and larger than the whole trajectory block. Accepted.
The pacing finding remains valid on training data and stays in the paper regardless.

---

## ADR-010 — 2026-07-26 — Verify feature COVERAGE, not just output format

**Context.** `submission.verify` passed 16/16 checks on a submission whose `main.py`
produced 110 of the model's 185 expected features. The feedback, trajectory and LO-position
blocks — including the single strongest block — were never wired into the inference path.
The missing 40% arrived as NaN, which LightGBM accepts silently, so the output was correctly
formatted, in range, and completely degraded. It would have consumed one of three weekly
attempts and produced a mysteriously bad leaderboard score.

**Decision.** `main.py` writes the **names** of the features it produced (names only — never
values or aggregates, which would violate the logging rule), and
`submission.verify::verify_feature_coverage` fails on any gap against the bundle's
`feature_cols`. `main.py` now computes every block, and window selection is shared with
training through `inference_lib.topk_spans` so the two cannot diverge.

**Alternatives rejected.** Relying on the parity unit tests — they compared *implementations*
of blocks that were wired in, and could not see a block that was simply absent from `main.py`.

**Consequences.** Adding a feature block now requires wiring it into `main.py` or the build
fails loudly. **Standing rule: format validity does not imply feature validity.**


---

## ADR-011 — 2026-07-27 — Forum rulings absorbed; unseen-objective regime measured

**Context.** The organizers answered a batch of clarification questions on the forum. Three
answers change our constraints materially, and one invalidates an assumption behind our CV.

**Decisions.**

1. **Hosted LLM APIs are now prohibited, not merely dispreferred.** *"Competition data can not
   be uploaded to an API."* `annotate.moves` must use the local vLLM backend. ADR-004 chose
   this for cost/risk reasons; it is now a rules requirement.
2. **Licence allowlist relaxed.** CC-BY-SA is explicitly acceptable, and the open-licensing
   requirement covers external models *and* data. Our Apache/MIT-only rule was over-strict.
3. **Added `evaluate.unseen_lo`.** *"Not every learning objective in the test set appears in
   the training set,"* but only **0.27%** of our CV validation rows exercise that regime — our
   headline score is measured almost entirely on seen objectives. The new task holds out whole
   objectives (moving entire sessions to validation so the transcript is never split) and
   scores only genuinely-unseen rows.
4. **Flagged multi-source distribution shift as the top generalizability risk.** *"The test set
   is not drawn entirely from Third Space Learning."* Our disfluency / `[unclear]` /
   `background`-role features are artifacts of one ASR pipeline and may not exist in another
   source or modality.

**Measured consequence (5 draws, 25% of objectives held out, ~8,500 rows scored per draw):**

| | log loss |
|---|---|
| model on unseen objectives | 0.59178 ± 0.01143 |
| prior (== `lo_only`, which degenerates to a constant) | 0.59982 ± 0.01224 |
| **transcript gain** | **−0.00804 ± 0.00260 (distinguishable from zero)** |

The absolute score is much worse than the seen-objective 0.54088 — expected, since the topic
prior is unavailable — but **the transcript contribution survives** at −0.0080 versus −0.0113
in the seen regime. That is the reassuring result, and it is exactly what the organizers say
they want: *"strong submissions will focus on identifying signals in the session transcripts,
rather than from the learning objective description alone."*

**Alternatives rejected.** Re-weighting CV to match a guessed unseen-objective rate — we do not
know the test proportion, and guessing it would trade a known-optimistic number for an
unknown-wrong one. Reporting both regimes separately is more honest.

**Consequences.** Two numbers are now reported, not one: the seen-objective score (0.54088 ±
0.00055) and the unseen-objective score (0.59178 ± 0.01143). The true leaderboard score sits
between them, at a mixing ratio only the organizers know.


---

## ADR-012 — 2026-07-27 — Ship no fitted sklearn estimator; pin to the container's version

**Context.** The first container smoke test exited 0 and produced valid output, but its log
carried three warnings:

> `InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version
> 1.9.0 when using version 1.8.0. This might lead to breaking code or invalid results.`

Our local scikit-learn was 1.9.0; the runtime image ships 1.8.0. It affected the Platt
calibrator and the TF-IDF vectorizer. Nothing crashed — "invalid results" would have been
silent, which is the failure mode that has already cost this project twice (the OOF clobbering
bug and the 40%-missing-features bug).

**Decision.** Two independent mitigations, because one was not enough:

1. **Pin `scikit-learn==1.8.0`** in `pyproject.toml` and `requirements-colab.txt` to match the
   container exactly, and retrain. `submission.verify::verify_sklearn_version` compares the
   manifest's build version against a declared `RUNTIME_SKLEARN` and fails on drift.
2. **Stop shipping fitted estimators where plain data will do.** The calibrator is now exported
   as numbers — Platt as `(coef, intercept)` applied via an explicit sigmoid, isotonic as
   threshold arrays applied via `np.interp` — and evaluated by arithmetic in
   `inference_lib.apply_calibration`. No sklearn object crosses the pickle boundary for
   calibration at all, so no future image change can affect it.

The TF-IDF vectorizer is still pickled; reconstructing its transform by hand carried more
reimplementation risk than the version pin removes. The pin plus the verify check covers it,
and a guard now fails the build if a pickled calibrator ever reappears.

**Alternatives rejected.** (a) Ignoring the warning — it explicitly says *invalid results*, and
we have no way to detect silent numerical drift in a leaderboard score. (b) Pinning alone —
leaves us exposed if the organizers update the image mid-competition.

**Consequences.** Retraining under 1.8.0 moved the score 0.54306 → 0.54360, inside the
±0.00055 noise band, and the winning calibrator flipped Platt → none — itself consistent with
the earlier finding that the Platt gain (+0.000058) was noise. Verify now runs 20 checks.
**Standing rule: read smoke-test warnings, not just exit codes.**


---

## ADR-013 — 2026-07-27 — Feature order comes from the model, and verify predicts before shipping

**Context.** Our first real submission scored **0.8006 log loss / 0.4933 AUROC, rank #229** —
AUC *below random*, and log loss worse than the 0.609 global prior, against a CV of 0.5436 /
0.7223.

The cause: ``build_submission._feature_cols`` read the column list from
``importance.parquet``, which ``model.gbdt`` writes **sorted by gain descending**. That is a
permutation of the training order in **179 of 181 positions**. ``main.py`` faithfully
reordered its computed columns to match, LightGBM read the DataFrame positionally, and every
feature was scrambled — the model saw `struct_n_utterances` where it expected `lo_prior_enc`,
and so on for 179 columns.

Nothing looked wrong. Predictions were confident, in range, correctly formatted, correctly
ordered against `submission_format.csv`, with all 181 feature names present. **All 20 verify
checks passed.** Reproduced locally in seconds once we ran the packaged `main.py` against
training data: log loss 0.80867, AUC 0.4738, mean prediction 0.4901 against a base rate of
0.7025.

**Decisions.**

1. **Feature order is read from ``booster.feature_name()``** — the model's own record, which
   cannot drift from the model by construction. All boosters are cross-checked against each
   other. ``importance.parquet`` is never an order source; the training order is *also*
   persisted separately to ``feature_order.json`` as an independent record.
2. **``main.py`` asserts the order at runtime** and raises rather than predicting if the
   bundle disagrees with any booster.
3. **``submission.verify`` gained two checks**, one structural and one behavioural:
   - ``verify_feature_order`` — bundle order must equal every booster's order.
   - ``verify_prediction_sanity`` — **runs the packaged ``main.py`` on real training data and
     requires log loss ≤ 0.50 and AUC ≥ 0.75**, plus a mean prediction within 0.10 of the
     training base rate. The model was trained on those rows; if it cannot fit them, the
     artifact is broken regardless of how well-formed its output is.

**Alternatives rejected.** Passing a numpy array in booster order — works, but hides the
mismatch instead of surfacing it. Trusting the name-based checks we already had — they
verified *presence*, and this bug was purely about *order*.

**Consequences.** After the fix, the same probe returns log loss **0.43927**, AUC **0.8291**,
mean **0.7349**. Verify now runs 23 checks. Three regression tests pin the behaviour,
including one that builds a real booster and asserts a permuted order is rejected.

**The lesson, stated plainly.** Our checks had drifted toward verifying *structure*: the file
exists, the columns are present, the rows align, the probabilities are in range. None of them
asked whether the thing *works*. Every verification suite needs at least one check that
exercises the artifact end-to-end against known answers.


---

## ADR-014 — 2026-07-27 — Independent review found 8 defects, incl. target-encoding leakage

**Context.** After the feature-order bug reached the leaderboard, an independent reviewer was
given `docs/REVIEW_PROMPT.md` and repo access. It found **eight** distinct defects. Two were
verified here from scratch before accepting any patch.

**Verified independently:**

1. **Target-encoding leakage (serious, affected every reported CV number).**
   `_fold_safe_lo_encoding` wrote a *single* `enc` array across all outer folds. Processing
   outer fold *k* overwrote the training-row encodings of every *other* fold — so fold 0's
   carefully leak-free validation encoding was clobbered during folds 1–4, replaced by one
   derived from data including fold 0's own labels. Only the last fold survived intact.
   Reproduced with a mutation test: flipping *only* fold 0's validation labels moved fold 0's
   own encoding by **−0.25**. A leak-free encoding cannot move at all.
   *Fix:* the encoding is computed per outer fold, so each booster gets its own fold-safe map.

2. **Subsample cohort mismatch (made `selftest.all` largely vacuous).**
   `cv.build` and the response-level blocks selected "the first N sessions" in
   *train_features row order*; session-level blocks selected them in *transcript filename
   order*. At N=400 the two cohorts **overlapped in 9 sessions — 2.2%**. Roughly 98% of
   self-test rows therefore had all-NaN session-level features, and the self-test passed while
   exercising almost nothing.
   *Fix:* cohort selection is centralized, and `build_matrix` now rejects a block that is
   absent for any row rather than letting LightGBM absorb the NaNs.

**Also fixed by the review** (each with a regression test): the verifier's prediction check
inspected a hardcoded archive rather than the one named by the caller; a stale calibrator
survived when "no calibration" won; the cross-row rules guard could return PASS when the flag
was enabled; unreadable transcripts produced finite tree predictions instead of the documented
prior fallback; OOF frames of equal length but different cohorts could be compared; and
`ensemble.blend` accepted an implicit experiment list.

**Consequences — reported numbers move.** With leakage removed, repeated-seed CV over 5 fold
assignments gives **0.54286 ± 0.00044** (was 0.54088 ± 0.00055) and a transcript contribution
of **−0.00934 ± 0.00040** (was −0.01132 ± 0.00066). The contribution remains real — 5/5 seeds,
CI excluding zero — but it is **smaller than previously reported**, and every earlier CV figure
in this repo was measured under leakage. Findings and the paper draft are updated accordingly.

**The lesson.** Three of our four self-inflicted bugs were caught by *our own* checks only
after they had already caused damage; these two were never caught at all. An outside reviewer
with an explicit brief about the failure *class* found in one pass what months of internal
checking had missed. Budget for external review before a deadline, not after a failure.
