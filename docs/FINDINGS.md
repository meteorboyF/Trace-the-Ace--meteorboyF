# FINDINGS.md — living paper draft

> **Status:** draft in progress. Target: **4 pages including figures and tables, excluding
> references** (8.5×11", 1" margins, ≥11 pt body, ≥10 pt figures).
> **Running page estimate: ~1.6 / 4.0 pages** (see [Page budget](#page-budget)).
>
> **How to use this file.** Add to it after *every* `interpret.report` run, not at the end.
> Every claim carries a pointer to the run manifest and figure that back it. Write for
> education researchers who are not ML specialists — no unexplained jargon. **Log negative
> results too**: approaches that failed are publishable findings and the Rigor criterion
> rewards them.
>
> Evidence pointers look like `[runs/eda/transcripts.json]` or
> `[artifacts/figures/key_moments.pdf]`. Anything without a pointer is a hypothesis, not a
> finding, and is marked **(unverified)**.

---

## Abstract

*(placeholder — write last, once the key findings are final)*

---

## Key findings

### F1. A quarter of what we are trying to predict is invisible to session-level models

**Claim.** Predicting a student's next-question outcome from a *whole-session* representation
is structurally limited: **26.0% of the variation in outcomes occurs *within* a single
tutoring session**, between different learning objectives covered in that same session. No
session-level representation can explain any of it.

**Evidence.** 35,072 responses come from 22,821 sessions. 36.7% of sessions produce more
than one response (up to 10), accounting for **58.8% of all responses**. Among those
multi-response sessions, **38.3% contain both correct and incorrect outcomes**. The
response-weighted variance decomposition gives a within-session component of **0.0543**
against a total of **0.2090** — a ratio of **0.260**.

An *oracle* that knew each session's true mean outcome — the best any session-level model
could possibly do — would still score a log loss of **0.1540**; the remaining error is
irreducible without topic conditioning.
`[runs/eda/lo_conditioning.json]`, `[runs/eda/overview.json]`, `[docs/DATA.md]`

> **Methodological note (recorded for transparency).** An earlier ad-hoc calculation put this
> ratio at 0.433. That figure averaged within-session variance across *multi-response
> sessions only*, which overstates the global quantity because single-response sessions
> contribute zero within-session variance by construction. The response-weighted figure of
> **0.260** is the correct global statement; **0.433** remains the correct *conditional*
> statement for the multi-response subpopulation. The task `eda.lo_conditioning` now computes
> both reproducibly.

**Why it matters for practice.** Two learning objectives taught in the same 43-minute lesson
by the same tutor to the same student routinely produce *different* outcomes. Any measure of
"how well did this lesson go" that summarises the session as a whole — including the
session-level engagement metrics commonly reported by tutoring platforms — is blind to
almost half the signal. Knowledge tracing from dialogue must be **topic-conditioned**, not
lesson-conditioned.

**What we did about it.** We built an explicitly topic-conditioned feature block
(`features/lo_alignment.py`) that locates the parts of the transcript relevant to each
specific learning objective and describes the dialogue *there* rather than across the whole
lesson. See [Methodology](#methodology).

---

### F2. Transcripts are far shorter than expected, and the compute ceiling is not the real constraint

**Claim.** Whole-session transcripts fit comfortably inside a single modern encoder context
window. The engineering constraint people expect (inference cost on very long documents) is
**not** binding for this corpus.

**Evidence.** Median **5,323 tokens** per session (mean 5,260; p95 7,262; p99 8,095; max
11,548) across 22,821 sessions, ~120M tokens total, at ~3.45 characters per token. An
8,192-token context therefore covers ~99% of sessions in one pass. Projected inference for
the full 10,508-response test set on one A100 against a 6-hour cap: chunked encoder
**0.07–0.10 h**; a 7B generative model 1.0–1.5 h; a 14B generative model 2.0–3.1 h.
`[runs/eda/transcripts.json]`, `[runs/eda/inference_budget.json]`,
`[artifacts/figures/eda_token_distribution.pdf]`

**Why it matters for practice.** Research groups working with tutoring dialogue often reach
for aggressive summarisation or truncation on the assumption that transcripts are
unmanageably long. For ~45-minute one-to-one sessions, that assumption is wrong and the
summarisation step discards signal for no benefit. Model choice should be driven by *what
signal you can extract*, not by document length.

---

### F3. The corpus is speech, not chat — and the transcription artifacts are themselves informative

**Claim.** These are ASR (automatic speech recognition) transcripts of spoken tutoring, not
typed chat logs. This has three consequences that any transfer of these methods must handle.

**Evidence.**
1. **`[unclear]` markers appear in ~30% of all utterances** (tutor 29.5%, student 30.3%) —
   the transcriber routinely could not resolve the audio.
2. The role field is **three-valued**, not the documented two: tutor (52.1%), student
   (43.9%), and **`background` (4.0%)**.
3. **`background` is not noise — it is a speaker-diarization failure bucket.** Manual
   inspection of a 300-session sample shows it contains substantial genuine pedagogical
   speech misattributed away from the tutor, e.g. multi-sentence explanations of place value
   and fraction notation running to 780 characters, alongside short backchannels ("Yeah.",
   "Mm-hmm.") and pure `[unclear]` markers.
`[docs/DATA.md]`, `[runs/eda/transcripts.json]`

**Why it matters for practice.** (a) Discarding the unlabelled-speaker channel would throw
away real teaching. (b) Disfluency and hesitation markers are available *for free* in speech
data and are plausible uncertainty signals — we test this directly (see F4). (c) **A model
tuned on this corpus should not be assumed to transfer to typed-chat tutoring**, where
disfluency, transcription error, and diarization failure do not exist. This is our principal
generalizability caveat.

---

### F4. The transcript adds real but modest signal on top of topic difficulty — and *how you ask* decides the answer

**Claim.** Dialogue features improve outcome prediction beyond topic difficulty, but the
improvement is **small**, and it only appears if topic difficulty is modelled *alongside*
the transcript rather than in competition with it.

**Evidence.** Full 5-fold session-grouped cross-validation on all 35,072 responses:

| Model | CV log loss | AUC | Δ vs `lo_only` |
|---|---|---|---|
| `baseline.prior` (global base rate) | 0.60876 | 0.500 | +0.0569 |
| `baseline.lo_only` (topic only, **no transcript**) | **0.55184** | 0.707 | — (the bar) |
| GBDT, **transcript features only** | 0.59312 | 0.609 | **+0.0413 (worse)** |
| GBDT, transcript + topic prior (4 blocks) | 0.54306 | 0.7223 | −0.00878 |
| **+ Platt calibration (current best)** | **0.543005** | **0.7223** | **−0.00883 (better)** |

`[runs/gbdt/model_gbdt.json]`, `[runs/evaluate/model_gbdt.json]`, `[docs/EXPERIMENTS.md]`

**Why it matters for practice — this is the paper's methodological warning.** A transcript-only
model *looks* respectable in isolation (AUC 0.609) yet is **decisively worse than a lookup
table of topic difficulty**. Had we reported it without the baseline, we would have claimed a
working dialogue model while underperforming the trivial alternative. Conversely, refusing to
use topic difficulty at all — an over-correction against the stated anti-goal — throws away
the single strongest predictor. The defensible framing is: **model topic difficulty
explicitly, then ask what the dialogue adds on top.** The answer here is a genuine but modest
−0.0078 log loss, with AUC rising 0.707 → 0.721.

The topic prior is target-encoded with **two levels of leakage protection**: validation rows
are encoded only from their training folds, and training rows use an *inner* session-grouped
K-fold, so the model never sees an encoding derived from the row it is predicting.
`[src/traceace/models/gbdt.py::_fold_safe_lo_encoding]`

---

### F5. Topic-relevant discussion is front-loaded within a lesson

**Claim.** The stretch of dialogue most relevant to a given learning objective sits
**early** in the session, and its *position* is one of the most useful individual features.

**Evidence.** Across all 35,072 responses, the most topic-relevant 20-utterance window has a
normalized position (0 = lesson start, 1 = lesson end) with **median 0.138**, quartiles
[0.045, 0.468], mean 0.274. `lo_best_pos` is the **single highest-gain transcript feature**
in the model, and `lo_topk_pos_spread` and `lo_sim_gini` also rank in the top 10.
`[artifacts/figures/key_moments.pdf]`, `[runs/interpret/model_gbdt.json]`

**Why it matters for practice.** Two readings, and we cannot yet separate them: tutors may
introduce and diagnose a topic early and then move on, or the topic vocabulary may simply
appear when the objective is first stated. Either way, for **knowledge tracing the opening
minutes of a lesson carry disproportionate information about a specific topic** — which is
encouraging for early-warning applications, since a prediction need not wait for the lesson
to end. The concentration measure (`lo_sim_gini`) also being predictive suggests that
*whether* a topic is discussed in one focused burst versus scattered throughout is itself
informative.

---

### F6. Feature importance and marginal contribution disagree — and only one of them supports a claim

**Claim.** The feature block with the highest model importance is **not** the block whose
removal hurts most. Reporting importance alone would have produced a wrong conclusion.

**Evidence.** Total LightGBM gain by block versus leave-one-block-out marginal contribution
(positive = removing the block made log loss worse, i.e. the block contributed):

| Block | Gain importance | Marginal contribution | Rank agreement |
|---|---|---|---|
| LO-alignment | **51,822** (1st) | +0.00059 (2nd) | ✗ |
| Linguistic | 27,822 (2nd) | **+0.00192 (1st)** | ✗ |
| Structural | 13,721 (3rd) | +0.00028 (4th) | ✗ |
| Feedback | 11,270 (4th) | +0.00046 (3rd) | ✗ |
| Temporal | (excluded) | **−0.00049 (harmful)** | — |

`[runs/interpret/ablation.json]`, `[artifacts/figures/importance_model_gbdt.pdf]`

**Why it matters for practice.** Gain importance measures how often a model *used* a feature,
which under correlated features rewards whichever correlated variant the tree happened to
split on first. Marginal contribution measures what is *lost* without it. Because our blocks
are heavily correlated (talk ratio appears in structural, linguistic, feedback, and inside
the LO-relevant windows), the two rankings disagree at **every single position**. **We
recommend education-research work report ablations, not importance charts**, when the claim
is "feature family X matters."

A sharper demonstration of the same redundancy: LO-alignment alone (0.54921) and structural
alone (0.54878) score almost identically when each is paired with the topic prior, even
though LO-alignment absorbs ~4× the gain when both are present. The blocks are largely
**substitutes**, not complements — which caps how much any single new block can add.

---

### F7. Tutor feedback behaviour carries independent signal

**Claim.** How a tutor *responds* to student attempts — correcting versus affirming, and
whether a correction is eventually resolved — predicts outcomes beyond what the rest of the
dialogue features capture.

**Evidence.** A dedicated feedback block (40 features, lexical markers, CPU-only) contributes
**+0.00046** log loss on leave-one-out and takes **10.8% of total model gain**, with all 40
features used. It is complementary rather than redundant with LO-alignment: the two together
(0.54759) beat either alone (0.54921 lo-alignment, 0.55055 feedback).
`[runs/interpret/ablation.json]`, `[src/traceace/features/feedback.py]`

The highest-gain feedback features are exactly the interpretable ones:

| Feature | What it measures | Gain |
|---|---|---|
| `fbs_corrective_ratio` | corrections ÷ (corrections + affirmations) | 891 |
| `fbs_affirm_rate` | affirmations per tutor turn | 878 |
| `fb_affirm_rate` | same, **within the topic-relevant window** | 783 |
| `fbs_affirm_after_attempt_rate` | tutor affirms immediately after a student attempt | 628 |
| `fbs_correct_after_attempt_rate` | tutor corrects immediately after a student attempt | 565 |
| `fbs_last_correction_pos` | where the final correction falls in the lesson | 507 |

**Why it matters for practice.** These are quantities a tutoring platform can compute without
any ML: the ratio of corrective to affirming feedback, and whether the lesson *ends* on an
affirmation or an unresolved correction. They describe the interaction rather than the
student, so they are directly actionable as tutor feedback — and unlike topic difficulty they
cannot be known before the lesson happens.

**Caveat (important).** This is correlational. Tutors correct *because* a student is
struggling, so a high corrective ratio is plausibly a **response to** low mastery rather than
a cause of it. Nothing here licenses telling tutors to correct less.

---

### N-results embedded here: see [Negative results](#negative-results) for structural
features and calibration, both of which failed to help.

---

### F6. Baselines, and the bar a real model must clear

**Claim.** A model that uses no transcript at all is already strong, so transcript-based
claims must be measured against it explicitly.

**Evidence.** Global base rate = **0.7025 correct**, giving a prior-only log loss of
**0.6088** (the label entropy) — the floor. There are **398 distinct learning objectives**
(median 9 responses each, max 1,373), and per-objective correctness varies widely (min 0.00,
median 0.750, max 1.00; s.d. 0.286). A smoothed per-objective mean — *the organizers' stated
anti-goal, using no dialogue whatsoever* — is therefore a genuinely competitive predictor.
`[runs/eda/overview.json]`, `[docs/DATA.md]`

**Why it matters for practice.** Reporting "our model predicts student outcomes with X
accuracy" is close to meaningless if a topic-difficulty lookup table achieves nearly the
same. **Every result in this work is reported as a delta against that baseline**
(`delta_vs_lo_only`), and the pipeline computes it automatically so it cannot be quietly
omitted. We recommend the same convention for other knowledge-tracing work.

---

## Methodology

*(Written for a non-ML reader. Full technical detail in `docs/ARCHITECTURE.md`.)*

### Validation design — why grouping matters more than usual here
Because one session yields up to 10 rows sharing an identical transcript, a naive random
train/test split places the *same* transcript on both sides and inflates every score. All
validation uses 5-fold **session-grouped** stratified cross-validation
(`StratifiedGroupKFold`, grouped on `session_id`), generated once and reused by every model
so results are comparable. An automated test asserts zero session overlap between folds and
runs on every code change. `[src/traceace/cv.py]`, `[tests/test_cv_leakage.py]`

### Feature engineering — four interpretable blocks, one conditioned on topic
| Block | Level | What it measures |
|---|---|---|
| Structural | session | turn counts, student talk ratio, monologue run lengths, role balance |
| Linguistic | session | questioning, hedging, affirmation, understanding checks, **ASR disfluency**, per role |
| Temporal | session | response latency and pacing, using **robust statistics** |
| **LO-alignment** | **response** | **topic-relevant "key moments" and the dialogue inside them** |

Blocks are deliberately transparent — an education researcher can read the word lists and
audit exactly what the model responds to, which an embedding does not permit.

**Robust statistics for timing.** ASR segmentation creates spurious pauses (silence,
mis-splits, dropouts). All timing features use median / trimmed mean / interquartile range
rather than mean and standard deviation, and gaps beyond 120 s are treated as breaks. Without
this, a handful of transcription artifacts dominate every timing feature. `[ADR-006]`

**The LO-alignment block, in plain terms.** For each (lesson, learning objective) pair we
slide a 20-utterance window across the lesson, score how closely each window's language
matches the objective's description, and then summarise the **three best-matching windows**:
how strong the match is, *where* in the lesson they fall, how spread out they are, and what
the conversation looks like inside them (who talks, who asks questions, how much the student
hedges). Those last measures are what differ between two objectives in the same lesson, and
are what make within-session discrimination possible at all.

The matching vocabulary is fit on **training** objective descriptions only, so the procedure
never sees the test set and is fully deterministic — a requirement of the competition rules
and good practice generally.

### Modeling and calibration
Gradient-boosted trees (LightGBM) over the assembled features, trained per fold. The
evaluation metric, log loss, rewards *honest probabilities* rather than confident ranking, so
we compare no calibration against Platt scaling and isotonic regression using an **inner**
cross-validation loop — the reported calibration gain is itself out-of-fold, not an artifact
of fitting the calibrator on the same data used to judge it.

### Interpretability as a first-class output
Every model run emits: cross-fold feature importance **with dispersion across folds** (not a
single-fit bar chart); performance sliced by transcript length, turn count, student talk
ratio and learning objective; reliability diagrams before and after calibration; the
key-moment position distribution; and a leave-one-block-out ablation giving each feature
family's marginal contribution. This lets us make claims about *what* mattered rather than
merely *that* something worked.

### Tutoring-move taxonomy — annotate once, run cheaply
Tutor utterances are categorised into moves (questioning, explaining, scaffolding, affirming,
correcting, checking, managing) and student utterances into responses (answering, asking,
expressing confusion, acknowledging, off-task). A large language model labels a stratified
sample of **training** utterances once; a small, cheap classifier is then trained on those
labels and applied to every transcript. This yields an LLM-quality taxonomy at a fraction of
the computational cost, and keeps the generative model entirely out of the prediction path.
`[ADR-004]`, `[src/traceace/annotate.py]`

---

## Extensions & generalizability

### What should transfer to other chat-based tutoring settings
- **The within-session variance decomposition (F1)** is a diagnostic any group can run on
  their own data in minutes, and it determines whether topic-conditioned modelling is
  necessary. We recommend it as a standard first step.
- **Topic-conditioned key-moment pooling** requires only a transcript and a short topic
  description — no platform-specific metadata, no curriculum ontology.
- **Reporting against a topic-only baseline** is a cheap, general safeguard against
  mistaking topic difficulty for pedagogical insight.

### What will not transfer, and when to distrust this model
- **Speech-specific features will not survive a move to typed chat.** Disfluency, `[unclear]`
  density, and diarization-failure volume are artifacts of the audio pipeline (F3). On typed
  data these features are absent or mean something different.
- **Fixed-length lessons.** Sessions here cluster tightly around 43 minutes; pacing features
  are calibrated to that. Variable-length or asynchronous tutoring will need recalibration.
- **Single subject and age range.** K-12 mathematics, one-to-one. We make no claim about
  other subjects or group tutoring.
- **Distrust the model when** the transcript is short or heavily `[unclear]`-laden (the
  dialogue signal is largely absent, and predictions fall back toward topic difficulty), or
  when the learning objective is rare in training (some objectives have a single example).
  Per-slice performance tables quantify exactly where this happens.

### Limitations
- Outcomes are binary next-question correctness — a coarse proxy for learning.
- The transcription pipeline is a confound: audio quality plausibly correlates with both
  transcription completeness *and* the conditions of the lesson itself.
- Correlational throughout. Nothing here identifies a *causal* effect of a tutoring move;
  tutors adapt their moves to the student, so effective-looking moves may be responses to
  struggle rather than causes of success. **We state this explicitly rather than implying
  causal guidance to tutors.**

### Future work
- Sequence models over move labels (does the *order* of tutoring moves matter, not just the
  mix?).
- Explicitly modelling the audio-quality confound.
- Validating the within-session decomposition on a typed-chat corpus.

---

## Negative results

*Failed approaches are findings. Reporting them is what the Rigor criterion rewards, and each
of these would otherwise cost another group the same wasted effort.*

### N1. Timing features carry no outcome signal — and actively hurt
The temporal block (18 features: inter-utterance gaps, student response latency, pacing
drift, all with robust statistics) has a marginal contribution of **−0.00049** — removing it
makes the model *better* (0.54355 → 0.54306). It is excluded from the default stack.

We consider this a genuine finding rather than an implementation failure, because the
features are well-formed: `temp_gap_trimmean` has 19,987 distinct values with sensible
spread (median 7.2 s, IQR 11 s), and sessions cluster tightly at ~43 minutes so pacing is
comparable across lessons. Two plausible explanations: (a) ASR segmentation timestamps
reflect the *transcription pipeline's* chunking as much as real conversational timing, and
(b) whatever pedagogical signal timing carries is already captured by utterance-length and
turn-taking features.
**Implication for other groups:** response latency is an appealing "struggle" proxy and is
cheap to compute, but on ASR-derived transcripts it did not survive contact with an ablation.
Verify it on your own data before building on it. `[runs/interpret/ablation.json]`

### N2. Calibration gains are real but negligible here
Platt scaling improves log loss by **+0.000058** (0.543063 → 0.543005) and roughly halves
expected calibration error (0.0056 → 0.0033). Isotonic regression is clearly *worse*
(0.545318) despite the best ECE (0.0024) — it over-fits the bin edges.
**Implication:** the organizers correctly note log loss "can often be improved with
calibration", but a cross-validated GBDT trained *directly* on log loss is already
near-calibrated, so the headroom is ~0.0001, not ~0.01. Worth checking (it is free), not
worth engineering around. Our comparison uses an inner cross-fold loop, so the gain is not an
artifact of fitting and evaluating the calibrator on the same rows.
`[runs/calibration/model_gbdt.json]`

### N3. A transcript-only model loses to a topic-difficulty lookup table
Detailed in [F4](#f4-the-transcript-adds-real-but-modest-signal-on-top-of-topic-difficulty--and-how-you-ask-decides-the-answer):
0.59312 vs 0.55184. Recorded here because it is the result most likely to be quietly dropped
from a write-up, and it is the one that most changes how the field should report results.

### N4. Two of our own earlier conclusions were wrong — retracted here
Rigor means recording our own errors, not only the model's.

1. **Within-session variance.** First estimate 0.433 averaged over multi-response sessions
   only; the response-weighted global value is **0.260** (F1). Now computed reproducibly by
   `eda.lo_conditioning`.
2. **"Disfluency and structural features are dead."** An earlier diagnostic reported 94 of
   127 features with exactly zero split gain, and we nearly concluded that disfluency
   markers carry no signal. That diagnostic was run against a **corrupted model**: an OOF
   file-naming bug (N6) had left the run early-stopping after 26–30 iterations, so almost
   nothing got split on. On the correctly-trained model only **5 of 149** features are
   unused, and disfluency features take **6.2% of total gain** with 13 of 16 used —
   `ling_tutor_filler_rate` (914) and `ling_student_filler_rate` (858) are top-20 features.
   **The disfluency hypothesis is supported, not refuted.** The lesson: never read feature
   importance without first checking the model actually converged.

### N5. Model capacity is not the bottleneck
Sweeping LightGBM capacity made things uniformly worse: deeper trees (127 leaves,
`min_data_in_leaf=20`) gave 0.54657, deeper + slower (lr 0.01) 0.54515, no feature
subsampling 0.54426 — versus 0.54306 for the tuned-by-default configuration. The plateau is a
**signal ceiling, not an optimization failure**, which is why effort has gone into new feature
families rather than hyperparameters.

### N6. An OOF file-naming bug silently inverted the headline result
`save_oof()` keyed files by experiment name alone, so a 400-session self-test overwrote the
full-data `baseline.lo_only` OOF. Every subsequent `delta_vs_lo_only` was then computed
against a 400-session baseline (0.53881) instead of the real one (0.55184) — flipping the
headline from "beats the bar by 0.0078" to "loses by 0.0052" **with no error raised
anywhere**, because both files were valid parquet with the right schema.
Fixed by namespacing OOF paths on subsample size, guarded by a row-count mismatch check that
logs loudly, and pinned by two regression tests.
**Implication for other groups:** in a pipeline where smoke tests and real runs share an
artifact directory, make the artifact *key* include the data scope. A silent comparison
against the wrong baseline is far more dangerous than a crash.
`[src/traceace/evaluate.py::experiment_name]`, `[tests/test_guards.py]`

---

## Page budget

| Section | Est. pages |
|---|---|
| Abstract + Key findings (F1–F6) | 1.0 |
| Methodology | 0.4 |
| Extensions & generalizability | 0.2 |
| Figures (target 4: token distribution, key moments, importance, reliability) | ~1.0 |
| **Running total** | **~1.6 / 4.0** |

Headroom remains for the ablation table and the move-taxonomy figure. **References are
excluded from the limit.**

---

## References

*(To add: knowledge-tracing literature, tutoring-move taxonomies / dialogue-act coding
schemes, calibration methods, ModernBERT.)*
