# Cross-row features — ANSWERED, do not post

**Status: RESOLVED 2026-07-09.** No question needs posting; it was already asked and
answered on the forum. Kept as the audit record.

---

## The ruling

Asked by `flzaccaria` (thread: *"Clarification on the independent-processing rule and
test-set session structure"*):

> could you clarify the scope of the rule that each test sample must be processed
> independently? Specifically: the test features file contains multiple rows (samples)
> belonging to the same session. Is it permitted to compute, for a given test sample,
> features derived from the structure of the test features file itself — e.g., the number of
> rows sharing that sample's `session_id`? This uses no labels and no transcript content from
> other samples, but it does read other rows of the test set.

Answered by `kwetstone` (organizer):

> **Yes, that approach would violate the rule that test samples must be processed
> independently. To make a prediction on any given test sample, the only input to your model
> drawn from the test should be that sample's metadata and transcript.**

That is our exact question, answered unambiguously. **Cross-row features are prohibited.**

## What this settles

- Our default (`features.allow_cross_row_aggregates: false`) was correct. Nothing we have
  built or shipped violates the rule.
- The **+0.00251 log loss is permanently forfeit**, not pending. `lopos_n_competing_los` was
  our second-highest-gain feature; it cannot be used at inference.
- The organizers also gave us a **clean, checkable principle**, which is more useful than a
  yes/no: *the only test-derived input is that sample's own metadata and its own transcript.*

## Re-audit against the stated principle (2026-07-27)

Every one of the 181 shipped features traces to a permitted source:

| Source | Count | Permitted? |
|---|---|---|
| this sample's own transcript (`struct_ ling_ temp_ fb_ fbs_ traj_`) | 172 | ✅ that sample's transcript |
| this sample's window within its own transcript (`lopos_` safe subset, `lo_`) | 8 | ✅ that sample's metadata × its own transcript |
| `lo_prior_enc` — lookup built from **training** labels | 1 | ✅ not drawn from test at all |

Fitted parameters are all training-only and therefore not "drawn from the test": the TF-IDF
vectorizer (training LO text), the Platt calibrator (training OOF), the per-objective prior
(training labels), the boosters, and the content PCA basis.

**One implementation note.** `main.py` groups test rows by `session_id` to avoid re-parsing
the same transcript file. That is I/O sharing, not information sharing: with the flag off, no
feature *value* is derived from any row other than the one being scored, and the transcript is
legitimately that sample's own input. `lo_position_features` is passed `[keep]` — this row's
spans only.

Enforced mechanically by `submission.verify::verify_no_cross_row_features`, which fails the
build if a cross-row feature reaches the shipped model.

## The research finding is unaffected

Objectives competing for a fixed lesson budget remains a valid, reportable result **on
training data** — where we may group freely. It stays in the paper (FINDINGS F9); it simply
cannot be a model input at inference. Prohibited-as-a-feature is not the same as
untrue-as-a-finding.

## Standing rule for future features

Before adding any feature, ask: *could I compute this for a single test row given only that
row's metadata and its own transcript, plus parameters fitted on training data?* If no, it is
prohibited. Re-run this audit whenever a block is added.
