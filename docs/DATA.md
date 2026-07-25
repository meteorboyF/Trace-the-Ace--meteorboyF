# DATA.md — measured data facts

> Numbers below are **measured from the real local files** (not the brief's estimates),
> via `eda.overview` and `eda.transcripts`. Regenerate with those tasks; JSON lives in
> `runs/eda/`. Last measured: 2026-07-25.

## Files (canonical, post-`data.ingest`)
| File | Rows (data) | Columns |
|---|---|---|
| `train_features.csv` | 35,072 | `response_id, session_id, learning_objective_id, learning_objective` |
| `train_labels.csv` | 35,072 | `response_id, is_correct` → canonicalized to `correct` |
| `submission_format.csv` (full test) | 10,508 | `response_id, probability` |
| `submission_format_smoke.csv` | 100 | `response_id, probability` |
| `train_transcripts.zip` | 22,821 session CSVs | `session_id, utterance_id, role, content, timestamp` |

Transcript zip: **22,821 files, 600.9 MB uncompressed**, integrity verified (`unzip -t`, exit 0).

## Responses & sessions
- **35,072 training responses** over **22,821 sessions**.
- **Responses per session:** mean 1.54, median 1, p90 3, p99 5, max 10.
  (One session → up to 10 learning-objective responses. This is exactly why CV must group by
  `session_id`, never `response_id` — see `CLAUDE.md`.)
- Test: **10,508 responses** (full), 100 (smoke). Smoke is drawn from training per the rules.
- No missing values in features or labels.

## Label
- Column is **`is_correct`** in the file (canonicalized to `correct` on load).
- **Positive (correct) rate = 0.7025.**
- **`baseline.prior` expected log loss = 0.6088** (the label entropy at p=0.7025). This is
  the reference floor — anything at or above it is worthless.
- Binary 0.0/1.0, no missing.

## Within-session label variance — why LO-conditioning is mandatory
Measured by `eda.lo_conditioning` → `runs/eda/lo_conditioning.json`.

| Quantity | Value |
|---|---|
| Sessions with >1 response | 8,364 (**36.7%** of sessions) |
| Responses in multi-response sessions | 20,615 (**58.8%** of all responses) |
| Multi-response sessions with **mixed** labels | 3,207 (**38.3%** of them) |
| Within-session variance (response-weighted) | **0.0543** |
| Total label variance | **0.2090** |
| **Within / total ratio** | **0.260** |
| Oracle session-only log loss (best possible for any session-level model) | **0.1540** |

**Interpretation.** 26.0% of outcome variance lives *inside* sessions, between different
learning objectives sharing one transcript. Session-level features assign identical values
to all responses in a session and therefore cannot address any of that 26%. This is the
measured justification for `features/lo_alignment.py` (see ADR-003).

> Note: **43.3%** is the corresponding ratio computed over multi-response sessions *only*
> (mean within-session variance 0.0905 / 0.2090). Both are correct; the response-weighted
> **0.260** is the right global figure and is what we report.

Responses-per-session counts: 1→14,457 · 2→5,640 · 3→1,904 · 4→581 · 5→168 · 6→48 · 7→17 ·
8→4 · 10→2.

## Learning objectives (relevant to the anti-goal)
- **398 unique learning objectives.** `learning_objective_id` ↔ `learning_objective` text are
  **1:1** (398 each), so the id is a clean categorical key for the text.
- ~88 responses per LO on average. This makes `baseline.lo_only` (per-LO mean correctness,
  smoothed) a **strong** baseline — the organizers' anti-goal is a real trap here, and every
  model report must beat this bar. See `CLAUDE.md` / brief §10.4.

## Transcripts — the length measurement that drives architecture
Measured with tiktoken `cl100k_base` as a license-clean token proxy (encoder token counts are
the same order of magnitude).

| Metric | p5 | p25 | **median** | mean | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| **tokens / session** | 3,071 | 4,467 | **5,323** | 5,260 | 7,262 | 8,095 | 11,548 |
| chars / session | 10,405 | 15,305 | 18,387 | 18,248 | 25,752 | 29,002 | 44,556 |
| utterances / session | 151 | 222 | 267 | 269 | 392 | 451 | 622 |
| duration (seconds) | 1,830 | 2,351 | 2,603 | 2,486 | 2,793 | 2,913 | 3,721 |

- **~3.45 characters per token.** Median session ≈ **5.3K tokens**, max ≈ 11.5K.
  **This is far shorter than the brief's 10–15K fear.** A single 8,192-token encoder window
  covers the p99 (8.1K); only ~1% of sessions need a second chunk.
- Sessions run ~40–45 min (median 43.4 min), tightly clustered — these are fixed-length lessons.
- **Total training corpus: ~120.0M tokens** across all sessions.

## Roles — three-valued, not binary
The brief's §2 says role ∈ {tutor, student}; the real data has **three**:
| role | utterances | share |
|---|---|---|
| tutor | 3,196,001 | 52.1% |
| student | 2,697,152 | 43.9% |
| **background** | 246,701 | 4.0% |

Total ≈ 6.14M utterances.

### `background` is a diarization-failure bucket, not a third speaker
Measured by `eda.roles` on a 300-session sample (81,129 utterances) → `runs/eda/roles.json`.
Programmatic verdict: **`diarization_failure_contains_real_speech`**.

| role | median chars | mean chars | max chars | % utts >200 chars | `[unclear]` per utt |
|---|---|---|---|---|---|
| tutor | 62 | 95.8 | 1,080 | 12.7% | 0.302 |
| student | 17 | 39.9 | 1,041 | 2.9% | 0.310 |
| **background** | 23 | 53.5 | **1,452** | **5.0%** | **0.388** |

**Interpretation.** `background` is bimodal: mostly short backchannels ("Yeah.", "Mm-hmm.")
and bare `[unclear]` markers, but **5% of it is long-form speech up to 1,452 characters** —
manual inspection shows genuine tutor explanations (place value, fraction notation,
end-of-lesson feedback) misattributed by speaker diarization. It also carries the *highest*
`[unclear]` rate, consistent with being the channel the transcriber struggled with.

→ **Do not drop it** (that discards real teaching), and do not treat it as a student/tutor
turn either. We keep it as its own role and expose its volume
(`struct_background_char_frac`) as a data-quality feature.

### These are ASR transcripts, not typed chat
`[unclear]` appears in **~30% of all utterances** across every role. Content carries
disfluencies, false starts, and redaction tokens (`[STUDENT_NAME]`, `[SPEAKER:x]`). Feature
code must not assume a two-role split, and the ASR nature is a first-class generalizability
caveat for the write-up: transfer to *typed* chat tutoring is non-trivial.

## Implications (carried into DECISIONS.md / architecture)
1. The **6-hour inference cap is not binding** at these lengths (see `eda.inference_budget`):
   a chunked encoder clears the full test set in ~6 minutes; even a 7B–14B generative LLM fits
   (1–3 h). Architecture is therefore driven by **signal quality and dev-time compute budget**,
   not the inference cap.
2. **Encoder-class model (ModernBERT-base, Apache-2.0, 8192 ctx) is the workhorse:** one window
   fits ~99% of sessions; frozen embeddings extracted once on L4 and cached forever.
3. `baseline.lo_only` will be strong (398 LOs, dense). Transcript features must demonstrably beat
   it — that delta is the headline research result.
