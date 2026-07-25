# CLAUDE.md — operating instructions for Trace the Ace

## ⛔ FIRST, EVERY TIME YOU RESUME
1. **Re-read [`docs/BRIEF.md`](docs/BRIEF.md) in full.** It is the authoritative v4 competition spec and
   requirements. This session is long; your context compacts. The brief is the source of truth,
   not chat history.
2. Then read [`docs/STATE.md`](docs/STATE.md) — current status, best score, next actions, units left.
3. Skim [`docs/DATA.md`](docs/DATA.md) for measured data facts before touching features/models.

## Golden rules (violating any of these can cost the competition)
- **Never commit competition data.** `.gitignore` + `.git/hooks/pre-commit` block it. Data lives in
  `data/raw/` locally and in Google Drive — never in git. See [`docs/COMPETITION.md`](docs/COMPETITION.md).
- **CV folds group by `session_id`, never `response_id`.** One session → many response rows.
  Splitting by response leaks transcripts across folds. `StratifiedGroupKFold(groups=session_id)`.
- **Log loss is the objective.** Clip predictions off 0/1. Calibration matters as much as ranking.
- **Every model report shows delta vs `baseline.lo_only`** (the organizers' anti-goal guard).
- **Progress bars hard-off in submission mode** (`PROGRESS_ENABLED=False`), enforced by `submission.verify`.
- **The submission must never print/log test-data info** (text, counts, sums, means, token totals).
- **Cheap-first ladder:** CPU baselines → frozen embeddings + GBDT → fine-tune only if exhausted.
  Idle attached GPUs are the #1 unit waste. Frozen embeddings extracted once on L4, cached forever.
- **Fail fast and loud on bad config.** Never silently default.

## License allowlist (external models/datasets)
Must permit **commercial use**. Apache-2.0 and MIT are safe (Qwen, Mistral, ModernBERT, DeBERTa).
**No** NC / CC BY-NC / research-only terms. Flag bespoke community licenses in
[`docs/EXTERNAL_ASSETS.md`](docs/EXTERNAL_ASSETS.md) as needing a forum question before building on them.
Winning solution must be MIT-licensable.

## Measured schema reality (differs from the brief's §2 prose — confirmed from the real files)
The brief's §2 describes the columns approximately; the actual files differ. `data.ingest` canonicalizes:
- `train_features.csv`: `response_id, session_id, learning_objective_id, learning_objective`
  (note the **extra `learning_objective_id`** column not mentioned in §2).
- `train_labels.csv`: `response_id, is_correct`  — the label column is **`is_correct`**, not `correct`.
  Canonicalize to `correct` internally.
- `submission_format*.csv`: `response_id, probability`.
- Transcript files: `session_id, utterance_id, role, content, timestamp`. `role` includes values
  **beyond** `tutor`/`student` (e.g. `background`); `timestamp` is **relative HH:MM:SS elapsed**, not an
  absolute datetime. Content contains ASR artifacts like `[unclear]` — these are **voice-transcribed**
  sessions, not typed chat. See [`docs/DATA.md`](docs/DATA.md) for the full enumeration.

## Environment
- Submission runtime: **Python 3.12 only**, CUDA 12.9, uv + PyTorch + vLLM, no network, 1×A100 80GB.
- Local dev Python: see [`docs/STATE.md`](docs/STATE.md) (managed to match 3.12 for parity).
- Colab Pro+ for GPU work. GitHub is source of truth; Colab is disposable. Drive holds large files.

## Where things live
- Package: `src/traceace/` — all logic. Notebook is a thin stable wrapper.
- Tasks: `src/traceace/tasks.py` registry. Run via `traceace.tasks.run("name", **kw)`.
- Config: `conf/base.yaml` (paths, seed, cv, unit-rate table — never hardcode rates).
- Docs: `docs/` — keep `STATE.md` current at end of every session, unprompted.
