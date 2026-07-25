# Reading the lesson, not the topic: what tutoring transcripts reveal about student mastery

*Working draft toward the Trace the Ace write-up (4 pages incl. figures, excl. references).
Numbers carry mean ± SD across 5 repeated fold assignments; see Methods.*

---

## Abstract

*(Placeholder — written last.)* We study whether a student's next-question outcome can be
predicted from the transcript of the tutoring lesson that preceded it, using 35,072 assessed
responses from 22,821 real one-to-one K-12 mathematics sessions. We report three findings.
First, roughly a quarter of outcome variance occurs *within* a single lesson, between
different objectives taught in it — a component no lesson-level representation can address,
and which we show requires conditioning features on the objective. Second, aggregate
transcript statistics have a low and heavily *substitutable* ceiling: feature families that
appear dominant by model importance are nearly interchangeable under ablation, and importance
and marginal contribution disagree at every rank. Third, the informative families are not the
ones usually reported by tutoring platforms: student **disfluency** and tutor **corrective
feedback** carry signal, while turn counts, talk ratio and response latency largely do not.
We report negative results alongside positive ones, and quantify the noise floor that makes
several plausible-looking effects unmeasurable at this sample size.

---

## 1. Key findings

### 1.1 A quarter of the target is invisible to lesson-level models

Each assessed response is a (lesson, objective) pair, and a single lesson often covers more
than one objective. In this corpus 36.7% of lessons produce more than one assessed response —
accounting for **58.8% of all responses** — and among those lessons **38.3% contain both
correct and incorrect outcomes**. The same transcript therefore has to explain two different
answers.

Decomposing outcome variance, **26.0%** is within-lesson (0.0543 of a total 0.2090). An
oracle told each lesson's true mean outcome, which is the best any lesson-level
representation could possibly do, still scores 0.1540 log loss rather than zero.

**Why this matters.** Engagement dashboards summarise a *session*. That summary is
structurally blind to a quarter of the variation in what students actually learned. Anyone
tracing knowledge from dialogue must condition on the topic being assessed, not the lesson as
a whole. The diagnostic is cheap — a variance decomposition on existing data — and we
recommend it as a first step before any modelling.

*(We report 0.260 rather than the 0.433 obtained when averaging over multi-response lessons
only; the latter conditions on a subpopulation and overstates the global quantity. Both are
computed by the released `eda.lo_conditioning` task.)*

### 1.2 The transcript helps, modestly, and only when topic difficulty is modelled alongside it

The natural baseline is a lookup table of per-objective difficulty that never reads the
transcript — precisely the shortcut the task's organizers name as an anti-goal. It scores
**0.55220 ± 0.00022** log loss.

A gradient-boosted model over transcript features *alone* scores **0.59312** — decisively
**worse** than the lookup table, despite a respectable AUC of 0.609 in isolation. Only when
topic difficulty is modelled *jointly* with the dialogue does the transcript add anything:
**0.54088 ± 0.00055**, a paired improvement of **−0.01132 ± 0.00066** (95% CI
[−0.01191, −0.01074]), with AUC rising 0.707 → 0.72576 ± 0.00085.

**Why this matters.** A transcript-only model reported without that baseline would look like
a working dialogue model while underperforming a spreadsheet. The opposite error — refusing to
use topic difficulty at all, to avoid the anti-goal — discards the strongest single predictor.
The defensible framing is to model topic difficulty explicitly and then ask what the dialogue
adds *on top*. Here the answer is real, replicable, and small.

### 1.3 Aggregate dialogue statistics are substitutable, and importance ranks mislead

Feature-importance charts are the usual currency of interpretability in this literature. They
are misleading here. Gain importance and leave-one-block-out contribution **disagree at every
rank**, because the blocks are heavily correlated — talk ratio appears in the structural
block, in the linguistic block, and again inside the topic-relevant window.

The clearest demonstration: LO-alignment alone (0.54921) and structural features alone
(0.54878) score almost identically when each is paired with the topic prior, even though
LO-alignment absorbs roughly four times the gain when both are present. They are **substitutes,
not complements** — which is also why adding any single new family moves the score so little.

**Recommendation.** Report ablations, not importance charts, whenever the claim is "feature
family X matters." Importance measures what a model *used*; only ablation measures what is
*lost* without it.

### 1.4 Order and time allocation dominate; disfluency follows; most families are unmeasurable

Paired leave-one-block-out across 5 fold assignments:

| Family | Marginal Δ (mean ± SD) | Seeds agreeing | Verdict |
|---|---|---|---|
| **trajectory** (order within the topic window) | **+0.00226 ± 0.00011** | 5/5 | real |
| **linguistic** (incl. disfluency) | **+0.00174 ± 0.00025** | 5/5 | real |
| lo-alignment (lexical similarity) | −0.00030 ± 0.00024 | 5/5 | significantly *hurts* |
| feedback | −0.00007 ± 0.00014 | 2/5 | unmeasurable |
| structural (turn counts, talk ratio) | +0.00006 ± 0.00030 | 4/5 | unmeasurable |
| temporal (latency, pacing) | +0.00005 ± 0.00047 | 4/5 | unmeasurable |

**Order is the single strongest family.** When things happened inside the topic-relevant
stretch — whether the student's utterances lengthened or shortened, whether corrections
cluster early or late, whether the episode *ended* on an affirmation or an unresolved
correction — beats every aggregate. This is a direct indictment of averaging: a student who
struggles then masters the topic and one whose fluency degrades into confusion produce
identical means and opposite outcomes.

**Time allocation is the second-strongest individual feature.** A plain count of how many
objectives shared the lesson is the second-highest-gain feature in the whole model, behind
only topic difficulty. Lessons are ~43 fixed minutes covering 1.54 objectives on average, so
objectives genuinely compete for time — a curriculum-pacing finding that needs no transcript
modelling to act on.

Two further families deserve mention, neither standard platform telemetry:

**Student disfluency.** These are voice-transcribed lessons, so hesitation is directly
observable. Filler rate, self-corrections, repetition and unresolved-audio density account for
**6.2% of total model gain** across 13 used features, with tutor and student filler rate both
in the top 20. Hesitation is a plausible, cheap uncertainty signal.

**Tutor corrective feedback.** How the tutor *responds* to an attempt — the corrective-to-
affirming ratio, whether affirmation follows immediately, whether a correction is ever
resolved — takes 10.8% of model gain, and `fbs_corrective_ratio` is a top-10 feature. But its
*marginal* contribution is −0.00007 ± 0.00014: once the trajectory block (scoped to the same
windows, and order-aware) is present, feedback adds nothing separable. We report it as
descriptively interesting and predictively redundant — an honest distinction that a gain chart
alone would have obscured.

**The caveat we insist on.** This is correlational. Tutors correct *because* a student is
struggling, so a high corrective ratio is plausibly a response to low mastery rather than a
cause of it. Nothing here licenses telling tutors to correct less.

### 1.5 Several plausible effects are simply unmeasurable at this sample size

At 35,072 rows the standard deviation of a paired log-loss difference across fold assignments
is ≈5×10⁻⁴, and the spread of the headline score across five fold assignments is 0.00105 —
**larger than most individual feature-family effects**. We initially excluded the timing
block on a single-seed reading of −0.00049 and had to reverse that decision: the reading was
inside the noise floor.

**Recommendation.** Report repeated-seed error bars on every CV comparison in this domain. A
single fold assignment is one draw from a distribution, and differences below ~10⁻³ on a
dataset this size should not be treated as measurements at all.

---

## 2. Methodology

**Validation.** One lesson yields up to ten responses sharing an identical transcript, so a
random split leaks the transcript across folds. All validation uses 5-fold session-grouped
stratified cross-validation, with an automated test asserting zero session overlap. Every
reported number is the mean ± SD across **five independent fold assignments**; block
comparisons are **paired** within each assignment so shared fold noise cancels.

**Features.** Six interpretable families: structural (turn counts, talk ratio), linguistic
(questioning, hedging, affirmation, disfluency, per speaker), temporal (latency and pacing,
robust statistics), LO-alignment (topic-relevant "key moments"), feedback (corrective response
to attempts), and trajectory (order-sensitive trends within the topic window). Word lists are
small and auditable by design.

**Topic conditioning.** For each (lesson, objective) pair we slide a 20-utterance window
across the lesson, score each window's relevance to the objective description, and summarise
the three best-matching windows — how strong the match is, *where* they fall, and what the
conversation looks like inside them. Those interior statistics are what differ between two
objectives in the same lesson, and they make within-lesson discrimination possible at all.
The matching vocabulary is fit on training objective text only, so the procedure is
deterministic and never sees the test set.

**Model and calibration.** Gradient-boosted trees, trained per fold. A capacity sweep
(deeper trees, slower learning rates, no feature subsampling) made results uniformly worse,
indicating a **signal ceiling rather than an optimisation failure**. Platt scaling is applied:
it improves log loss by only +0.000058, which we do **not** count as a score gain, but it
roughly halves expected calibration error (0.0056 → 0.0033), and honest probabilities are the
point of the metric.

---

## 3. Extensions and generalizability

**What should transfer.** The within-lesson variance decomposition is a few lines of code on
any (session, objective, outcome) table and determines whether topic conditioning is necessary
at all. Topic-conditioned window pooling needs only a transcript and a short topic description
— no platform-specific metadata or curriculum ontology. Reporting against a topic-only
baseline is a cheap, general guard against mistaking topic difficulty for pedagogical insight.

**What will not transfer.** The disfluency family is an artifact of *speech*: these are ASR
transcripts, with unresolved audio in ~30% of utterances and a third "unattributed" speaker
channel produced by diarization failure that nevertheless contains genuine teaching. On typed
chat those features are absent or mean something different. Lessons here are a fixed ~43
minutes and one-to-one in K-12 mathematics; pacing features are calibrated to that.

**When to distrust the model.** Short transcripts, heavily unresolved audio, and objectives
that are rare in training all push predictions back toward topic difficulty. Per-slice tables
quantify where this happens.

**Limitations.** Binary next-question correctness is a coarse proxy for learning. Audio
quality is a confound: it plausibly correlates with both transcription completeness and the
conditions of the lesson itself. Everything here is correlational.

**Future work.** Sequence models over tutoring-move labels (does the *order* of moves matter,
not just the mix?); explicit modelling of the audio-quality confound; and replication of the
within-lesson decomposition on a typed-chat corpus.

---

## 4. Negative results

Reported because they cost us time and would cost others the same, and because a family that
fails here may still work elsewhere.

1. **Timing features carry no measurable signal** (+0.00005 ± 0.00047) — and our first attempt
   to *exclude* them was itself unjustified (1.5). The honest verdict is "unmeasurable", not
   "absent".
2. **Structural features are redundant**, not useless (+0.00006 ± 0.00030): nearly
   interchangeable with richer families (1.3). "Student talk ratio" as a standalone quality
   metric is largely subsumed by *what* is said.
2b. **Redundancy is contextual.** LO-alignment measured +0.00059 alone and −0.00030 ± 0.00024
   once trajectory was added. Nothing about the block changed; its neighbourhood did. One
   ablation over one feature set is a snapshot, not a property of the features.
3. **Calibration headroom is ~10⁻⁴, not ~10⁻²** on a model already trained on log loss.
4. **A transcript-only model loses to a topic lookup table** (1.2).
5. **Model capacity is not the bottleneck** — every capacity increase hurt.
6. **Two of our own earlier conclusions were wrong and are retracted**: an inflated variance
   ratio (1.1), and a "disfluency is dead" reading produced by a corrupted run in which the
   model had early-stopped after ~26 iterations. Correctly trained, only 5 of 149 features go
   unused and disfluency is among the strongest families. *Never read feature importance
   without first checking the model converged.*

---

## References

*(To add: knowledge tracing; dialogue-act / tutoring-move coding schemes; calibration;
sentence-embedding retrieval.)*
