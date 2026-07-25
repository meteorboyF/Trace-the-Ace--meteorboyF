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

## ADR-009 — 2026-07-26 — Ship without cross-row feature inputs pending a rules ruling

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
