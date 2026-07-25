# COMPETITION.md — the rules that constrain engineering

Condensed from the authoritative spec in [`BRIEF.md`](BRIEF.md) §2. This file holds only
what changes an engineering decision. When in doubt, the brief wins.

## What we are predicting
Given a student–tutor session transcript plus a learning-objective description, predict
**P(student answers the next question on that topic correctly)**. One sample = one
(session, learning objective) pair; a session may yield several samples.

## Metric
**Log loss**, lower is better. ROC AUC is displayed but **does not affect rank**.
Log loss punishes confident-and-wrong, so calibration is worth as much as ranking —
the organizers say so explicitly. Never emit exactly 0 or 1 (infinite loss); we clip at
`predict_clip_eps` from `conf/base.yaml`.

## How placement actually works
Leaderboard rank is a **gate**: top 15 are invited to submit a write-up, and final prizes
combine leaderboard standing with write-up quality — Relevance 35%, Generalizability 35%,
Communication 15%, Rigor 15%. **~70% of the judged component rewards transferable
insight**, not decimals of log loss. This is why `interpret.report` is a first-class output
and `FINDINGS.md` is a living paper draft.

**Named anti-goal:** predicting correctness from the learning objective's inferred
difficulty *without* reference to the transcript. Structurally guarded: `baseline.lo_only`
implements exactly that anti-goal, and every report prints `delta_vs_lo_only`.

## Deadlines
- Model submissions close **2026-08-27 23:59 UTC**
- Write-up (top 15 only) closes **2026-09-15**

## Submission format — fatal-mistake territory
- `submission.zip` with **`main.py` at the archive ROOT** (no wrapping folder).
- `main.py` writes `submission.csv` **beside itself**, columns `response_id` (str) and
  `probability` (float in [0,1]), matching `submission_format.csv` **exactly** (row set
  *and* ordering).
- At runtime the working dir has read-only `data/` containing `submission_format.csv`,
  `test_features.csv`, `test_transcripts/{session_id}.csv`.

## Runtime limits
| Constraint | Value |
|---|---|
| Python | **3.12 only** |
| Stack | uv + PyTorch + vLLM, CUDA 12.9 |
| Network | **none** — all weights vendored in the zip |
| Hardware | 1× A100 80 GB, 24 vCPU, 220 GB RAM |
| Zip size | ≤ 60 GB (we enforce 55 GB) |
| Full run | ≤ **6 h** (we enforce a 4.5 h projection) |
| Smoke run | ≤ **10 min** |
| Logging | **≤ 500 lines × 500 chars** (we enforce 400 lines) |

Packages must already exist in their image; additions need a GitHub issue on the runtime
repo. **Current status: no additions needed** — see [`EXTERNAL_ASSETS.md`](EXTERNAL_ASSETS.md).

## Disqualification risks — treat as hard constraints
1. **Never print or log anything about the test data** — no excerpts, no learning-objective
   text, and no aggregates (counts, sums, means, token totals). `submission.verify`
   statically scans every shipped `.py` and rejects non-literal emissions.
2. **Each test sample processed independently.** No pseudo-labeling, no unsupervised
   learning on the test set, no information shared across test samples. Training with
   different or absent test data must produce identical weights and fitted parameters.
   (This is why the LO TF-IDF vectorizer is fit on *training* LO text only.)
3. **Progress bars off** in submission mode — tqdm's carriage returns can blow the log cap.

## Submission budget
**Three full submissions per week.** Smoke tests, cancelled and failed jobs do **not**
count. ~15 real attempts remain. Order of operations the organizers ask for:
local `just test-submission` → smoke environment → full submission.
This is why a loud `submission.verify` is worth more than a clever model.

## Licensing
External models/datasets must be public and commercially licensed (no NC/research-only).
Winning solutions are MIT-licensed; winners complete a Winning Model Documentation
Template. Solo entry, one account. Public code sharing is permitted and auto-MIT-licensed;
private sharing outside a team is prohibited.

## Write-up format (plan for it now)
PDF, **max 4 pages including figures and tables, excluding references**, 8.5×11", 1"
margins, ≥11 pt body / ≥10 pt figures, ≥ single spacing. Sections: Key findings,
Methodology, Extensions & generalizability. Top write-ups are invited to develop into a
full academic paper.
