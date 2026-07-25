# Trace the Ace — meteorboyF

Student-outcome prediction from K-12 tutoring transcripts, for the DrivenData
["Trace the Ace"](https://www.drivendata.org/) competition (K-12 AI Infrastructure Program,
prize from the National Tutoring Observatory).

**Task.** Given a student–tutor session transcript and a learning objective, predict the
probability the student answers the next assessment question on that topic correctly.
**Metric:** log loss (calibration matters as much as ranking). Solo entry.

> **Resuming work?** Read [`CLAUDE.md`](CLAUDE.md) → [`docs/BRIEF.md`](docs/BRIEF.md) →
> [`docs/STATE.md`](docs/STATE.md) first, in that order. The brief is the authoritative spec.

## Why this repo is shaped the way it is
- **All logic lives in `src/traceace/`.** The Colab notebook is a thin, stable wrapper that
  `git pull`s and re-imports. GitHub is the source of truth; Colab is disposable.
- **Compute is a hard constraint** (733 units, ~5 weeks). Tasks declare a runtime *tier* and a
  guard refuses to run CPU work on a paid GPU. Cheap-first ladder: CPU baselines → frozen
  embeddings + GBDT → fine-tuning only if exhausted.
- **The write-up is a first-class deliverable** (publication bonus). `interpret.report` emits
  publication-quality research artifacts alongside every model; `docs/FINDINGS.md` is a living
  paper draft.

## Layout
```
src/traceace/        package: config, paths, runtime, budget, staging, tasks, features, models, ...
conf/base.yaml       paths, seed, cv, unit-rate table (never hardcoded)
docs/                BRIEF, STATE, DATA, ARCHITECTURE, RUNBOOK, DECISIONS, FINDINGS, ...
notebooks/           Trace_the_Ace_Runner.ipynb (stable Colab wrapper)
submission/          built submission.zip staging
tests/               leakage, format round-trip, tier guard, path guard, ...
```

## Quickstart (local, CPU)
```bash
# Environment is a uv-managed Python 3.12 venv (matches the runtime).
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# Run the full pipeline end-to-end on a tiny real-data subsample (<5 min, CPU):
.venv/bin/python -m traceace selftest.all

# Lint + type + tests:
.venv/bin/ruff check . && .venv/bin/mypy && .venv/bin/pytest
```

## Running a task
```python
import traceace

traceace.configure(repo_dir=".", drive_root=None)  # local mode
traceace.tasks.run("eda.overview")
traceace.tasks.run("model.gbdt", subsample=2000)
```

## Data & compute safety (do not defeat these)
- Competition data is **never** committed. `.gitignore` + `.git/hooks/pre-commit` enforce it.
  Data lives in `data/raw/` locally and in Google Drive only.
- CV folds group by `session_id` (never `response_id`) — see [`CLAUDE.md`](CLAUDE.md).
- The packaged submission must never print test-data info and must disable progress bars.

License: MIT (required for winning solutions).
