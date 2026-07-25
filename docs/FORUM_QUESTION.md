# Forum question — ready to post

**Status:** DRAFT, awaiting posting by the operator.
**Blocking:** whether `lopos_*` cross-row features may be used at inference.
**Cost of assuming "no":** +0.00251 CV log loss (measured, 5 fold assignments).
**Cost of assuming "yes" and being wrong:** disqualification.

Until an answer arrives we ship the SAFE variant (`features.allow_cross_row_aggregates:
false`, enforced by a `submission.verify` check).

---

## Post this to the competition forum

> **Subject: Does deriving a feature from session grouping within `test_features.csv` violate independent processing?**
>
> Hi — I'd like to check a feature-engineering approach against the independent-processing
> rule before I build further on it, because I can't tell from the wording whether it's
> permitted and the downside is disqualification rather than a bad score.
>
> **The rule as I read it:** "each test data sample should be processed independently …
> precludes using information gathered across multiple test samples as feature inputs."
>
> **What I'm doing.** Each sample is a (session, learning objective) pair, and one session
> can appear in several rows of `test_features.csv` because several objectives were
> assessed in that lesson. I compute a feature that is simply **the number of rows in
> `test_features.csv` sharing this row's `session_id`** — i.e. how many objectives were
> assessed in the same tutoring session — plus a couple of derived positional features
> (this objective's ordinal position among that session's objectives).
>
> The motivation is pedagogical rather than statistical: lessons are a fixed ~43 minutes,
> so objectives compete for a fixed time budget, and how many share the lesson is
> informative about how much attention each one got.
>
> **Why I think it might be allowed:**
> - `session_id` is **provided metadata in the feature file**, not something I infer from
>   the test set or learn across samples.
> - I'm not fitting anything on test data, not pseudo-labeling, and not using any *label*
>   or *prediction* from another row. The parameters of my model are identical regardless
>   of what the test set contains.
> - Running my pipeline on a different subset of test rows changes no fitted parameter —
>   only this one count, which is arguably part of the provided input description of a
>   lesson.
>
> **Why I think it might not be allowed:**
> - Computing it literally requires reading other rows of `test_features.csv`, which is the
>   plain reading of "information gathered across multiple test samples as feature inputs."
> - The value depends on test-set composition: if you scored a subset of rows, the same
>   sample would get a different feature value. That seems like exactly the property the
>   rule is designed to exclude.
>
> **My questions:**
> 1. Is computing per-session response counts from `test_features.csv` (grouping test rows
>    by the provided `session_id`) a violation of the independent-processing requirement?
> 2. If it is not allowed in general, does it become allowed if the same quantity is derived
>    from the **transcript file alone** rather than from the feature file?
> 3. More generally: is the intended boundary "no *fitting* on test data" (so provided
>    metadata may be grouped), or the stricter "each row's features must be computable from
>    that row plus its own transcript in isolation"? A statement of the principle would let
>    me audit the rest of my features myself.
>
> I've defaulted to **excluding** these features pending an answer, so nothing I submit in
> the meantime relies on them. Thanks very much.

---

## Audit performed while drafting this

Every feature in the pipeline was traced to its inputs. The **only** cross-row argument
anywhere in the codebase is `all_lo_spans` in `inference_lib.lo_position_features`, which
feeds exactly four features:

| Feature | How it is derived | Status |
|---|---|---|
| `lopos_n_competing_los` | count of rows sharing this `session_id` | ⚠️ cross-row |
| `lopos_ordinal` | rank of this objective among the session's objectives | ⚠️ cross-row |
| `lopos_ordinal_frac` | same, normalized | ⚠️ cross-row |
| `lopos_overlap_with_others` | overlap of this objective's window with the others' | ⚠️ cross-row |

**Everything else is independent by construction** — each remaining feature is a function of
(this row's learning-objective text, this session's own transcript, parameters fitted on
training data only):

- `struct_*`, `ling_*`, `temp_*` — this session's transcript only
- `lo_*` (alignment) — this row's objective text vs this session's windows; the TF-IDF
  vocabulary is fit on **training** objective text
- `fb_*` / `fbs_*`, `traj_*` — this session's transcript, scoped to this row's objective
- remaining `lopos_*` (`centre_pos`, `start_pos`, `end_pos`, `utt_share`, `minute_share`,
  `duration_s`, `gap_to_session_end_*`) — this row's window within its own transcript
- `lo_prior_enc` — a lookup table built from **training** labels only
- `cont_*` (pending) — this session's window embeddings; PCA basis fit on **training** data

Grouping test rows by `session_id` inside `main.py` for **file-reading efficiency** (so each
transcript is parsed once) does not itself create a dependency: with the safe flag set, no
feature *value* is derived from any row other than the one being scored.

Enforced mechanically by `submission.verify::verify_no_cross_row_features`, which fails the
build if a cross-row feature reaches the shipped model while the flag is off.
