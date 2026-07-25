# Claude Code Prompt v4 — Trace the Ace

> Self-contained. Paste the whole thing into Claude Code from inside the repo folder.
> Supersedes v1–v3.

---

## 0. What is in this folder

This directory contains **exactly five files and nothing else** — no documentation, no README, no spec files. Do not go looking for any:

```
submission_format_5muR4s3.csv     1,224 B
submission_format_ZQLcKx7.csv     123 KB
train_features_TMQTWsB.csv        2.3 MB
train_labels_44ujmj2.csv          411 KB
train_transcripts.zip             576 MB
```

**All competition specification is embedded in §2 of this brief.** That section is authoritative — it is transcribed from the organizers' problem description and code-submission pages. There is no other source in this folder.

## 1. FIRST ACTIONS — in this order, before writing any other code

1. **Protect the data.** The GitHub repo is public and the competition rules prohibit redistributing the Data. Before any `git add`: write `.gitignore` covering `data/`, `artifacts/`, `runs/`, `*.zip`, `*.csv`, `*.parquet`, `*.pt`, `*.bin`, `*.safetensors`, `submission/assets/`, `train_*`, `test_*`, `submission_format*`. Install a pre-commit hook hard-blocking any file over 1MB and anything matching those patterns. Run `git status`, confirm nothing data-shaped is staged, **and report to me before continuing**.
2. **Persist this brief.** Save this entire document verbatim to `docs/BRIEF.md`. Add a line to `CLAUDE.md` telling you to re-read it whenever you resume. This session will be long and your context will compact — the requirements must survive as a file, not as chat history.
3. **Verify the environment and report:**
   - Is this a git repo with the correct remote? If not, `git init` and set origin. Do not commit until `.gitignore` exists.
   - Local Python version. The self-test suite in §7 must run here. If it isn't 3.12, say so and propose how to proceed.
   - `unzip -t train_transcripts.zip` — report session-file count and total uncompressed size. This download was recently interrupted, so confirm integrity before building anything that depends on it.
4. Move the raw files into `data/raw/` locally so the repo root is clean.
5. Do not try to fetch the competition platform pages — they are behind authentication. The runtime repo (`https://github.com/drivendataorg/tutoring-outcomes-runtime`) is public and fetchable.
6. I am competing **solo**, not as a team.

---

## 2. Competition specification (authoritative)

### Overview

**"Trace the Ace"** — K-12 AI Infrastructure Program, hosted by DrivenData, prize supplied by the National Tutoring Observatory.

Predict student quiz performance from tutoring lesson transcripts. Given a student–tutor transcript, predict whether the student answered the next question on the same topic correctly. The stated research aims are to understand which tutoring strategies best support learning, and how to perform student knowledge tracing from dialogue alone — monitoring understanding without costly testing.

Data is real student–tutor conversations paired with learning outcomes, collected with **Third Space Learning (TSL)** and **Eedi**.

### Deadlines

- Model submissions close **2026-08-27 23:59 UTC**.
- Write-up (top 15 teams only) closes **2026-09-15**.

### How winning works — this shapes the entire build

The top 15 teams on the final leaderboard are invited to submit a solution write-up. **All prizes are based on a combination of leaderboard performance and write-up quality.** Leaderboard rank is a gate; the write-up decides placement.

Write-up scoring:

- **Relevance 35%** — does it surface meaningful, actionable insight about (a) what makes tutoring effective and (b) student knowledge tracing? Does it give actionable guidance to other researchers?
- **Generalizability 35%** — would the approach and conclusions transfer to other chat-based tutoring setups, and is that demonstrated?
- **Communication 15%** — accessible to education researchers who are not ML experts.
- **Rigor 15%** — appropriate and correctly implemented methodology.

The organizers direct participants to focus on uncovering insight rather than solely maximizing performance, and name three target directions:

- What different feature-engineering approaches reveal, and how to extract generalizable features from high-dimensional dialogue data.
- Identifying **types of tutoring moves**, and which moves are most effective.
- Many transcripts are very long — identifying **"key moments"** that reveal student understanding.

They also name an explicit **anti-goal**: predicting correctness from inferred difficulty of the learning objective description without reference to the session transcript. Strong submissions find signal in the transcript. Guard against this structurally.

Top write-ups are invited to develop into a full academic paper for publication.

Write-up format: PDF, **maximum 4 pages including figures and tables but excluding references**, 8.5×11" with 1" margins, minimum 11pt main text / 10pt figures and tables, minimum single-line spacing. Expected sections: Key findings, Methodology, Extensions & generalizability.

### Data

Each sample is a student response — a unique combination of tutoring session and learning objective. **A single session may correspond to multiple samples** if multiple learning objectives were completed.

`train_features.csv` — response-level metadata:

- `response_id` (str) — unique sample identifier
- `session_id` (str) — links to a transcript file
- `learning_objective` (str) — short description of the objective being tested

`train_transcripts/` — one CSV per session, filename is the `session_id`. Columns:

- `session_id` (str)
- `utterance_id` (str) — unique within session
- `role` (str) — `tutor` or `student`
- `content` (str) — utterance text
- `timestamp` (datetime)

Labels — `response_id` (str), `correct` (float: 0.0 incorrect, 1.0 correct).

External datasets and pretrained models are permitted provided they are publicly available and openly licensed for commercial use.

### Metric

**Log loss**, averaged over observations:

```
LogLoss = −( y·log(p) + (1−y)·log(1−p) )
```

Lower is better. Log loss penalizes confident-but-wrong predictions and rewards well-calibrated probabilities. The organizers explicitly note that log loss can often be improved with calibration. **ROC AUC is displayed on the leaderboard for reference but does not affect position.**

### Submission — code execution challenge

You submit a packaged model and inference code, not predictions.

`submission.zip` must contain **`main.py` at the root level of the archive** — no wrapping folder. When unzipped, `main.py` must be in the folder where you unzip. Assets (model weights etc.) go alongside, conventionally in `assets/`.

At inference, the working directory contains:

```
├── data/                        # READ-ONLY
│   ├── submission_format.csv
│   ├── test_features.csv
│   └── test_transcripts/{session_id}.csv
├── main.py
└── <your assets>
```

`test_features.csv` and `test_transcripts/` exactly match the training structure. `main.py` must write `submission.csv` into the directory containing `main.py`, with columns `response_id` (str) and `probability` (float in [0,1]), matching `submission_format.csv` exactly.

### Runtime limits — design against these

- **Python 3.12 only.** No other languages or Python versions.
- Runtime built with **uv, PyTorch, and vLLM, backed by CUDA 12.9**. Packages must be pre-installed in their image; additions require a GitHub issue on the runtime repo.
- **No network access** inside the container. All weights and dependencies must be vendored into the zip.
- **1× NVIDIA A100 with 80GB VRAM**, 24 vCPUs (AMD EPYC 7V13), 220GB RAM.
- Zip archive **must not exceed 60GB**.
- Full submission must complete in **6 hours or less**. Smoke tests in **10 minutes or less**.
- **Logging is capped at 500 lines, 500 characters per line.**
- No root filesystem access.
- All code should run within the GPU environment even if computation happens on CPU.

### Submission rules that carry disqualification risk

- **The submission must not print or log any information about the test dataset** — including transcript excerpts, learning objective descriptions, or aggregations such as sums, means, or token counts. The organizers state this may be grounds for disqualification under the requirement that test samples are processed independently.
- Each test sample must be processed **independently**. No pseudo-labeling, no unsupervised learning on the test set, no information shared across test samples. Running training with the same training data but different or absent test data must produce identical weights and fitted parameters. Solutions must run inference on new test data without retraining.
- **Three full submissions per week.** Smoke tests, cancelled jobs, and failed jobs do not count against the limit. With ~5 weeks left that is roughly 15 real attempts.
- Testing order the organizers ask for: test locally with `just test-submission` in the runtime repo → smoke test environment → full submission.
- The smoke environment uses a sample of **100 responses drawn from the training set**, with the same file structure.
- Winning solutions must be MIT-licensed; supporting software must be open-source and commercially usable. Winners complete a Winning Model Documentation Template.
- Solo entry, one account. Private code sharing outside a team is prohibited; public sharing is permitted and auto-MIT-licensed.

---

## 3. Data inventory — normalize these on ingest

Downloaded files carry random browser suffixes. Write `data.ingest` to detect files by prefix **and content shape**, not exact name, and rename to canonical form:

| Downloaded | Canonical | Size | Notes |
|---|---|---|---|
| `train_features_TMQTWsB.csv` | `train_features.csv` | 2.3 MB | |
| `train_labels_44ujmj2.csv` | `train_labels.csv` | 411 KB | |
| `submission_format_ZQLcKx7.csv` | `submission_format.csv` | 123 KB | **full test set** |
| `submission_format_5muR4s3.csv` | `submission_format_smoke.csv` | 1,224 B | **smoke set, ~100 rows** |
| `train_transcripts.zip` | `train_transcripts.zip` | 576 MB | |

Distinguish the two format files by row count, not filename — confusing them would waste one of three weekly submissions.

**Verify these first-order estimates against the real files and record actuals in `docs/DATA.md`:**

- ~25–28K training responses.
- ~10K test responses (123KB at ~12 bytes/row).
- 576MB compressed transcripts → likely 2.5–3GB raw text.

**The token distribution is the single most important measurement in this project and must be resolved before any architecture is chosen.** Compute exact per-session character and token distributions, then project inference cost:

```
tokens_per_sample × n_test_samples ÷ throughput_tokens_per_sec   vs.   6-hour cap
```

If transcripts average ~10–15K tokens across ~10K test samples, that is on the order of 10⁸ tokens on one A100. A chunked encoder (ModernBERT / DeBERTa class) at ~150K tok/s clears it in minutes. A 7B generative model doing full-transcript prefill at ~10K tok/s lands near 4 hours — under the cap but with no margin, and one bad throughput assumption puts you over.

Make **`eda.inference_budget`** a real task printing a go/no-go table for candidate architectures against measured token counts. Architecture choice must follow from this arithmetic, not from vibes.

---

## 4. Compute budget — a hard constraint

733 compute units, ~5 weeks, Colab Pro+. Approximate burn rates — **verify against Colab's live Resources panel and keep the table in `conf/base.yaml`, never hardcoded:**

| Runtime | ≈ units/hr | ≈ hrs from 733 | Use for |
|---|---|---|---|
| CPU + High RAM | ~0 | unlimited | I/O, EDA, features, GBDT, calibration, blending, reporting, packaging |
| T4 | ~2 | ~350 | small encoder tests, GPU sanity checks |
| L4 | ~5 | ~145 | embedding extraction, mid-size fine-tunes |
| A100 80GB | ~12 | ~60 | large fine-tunes, final submission timing validation |
| H100 | higher | fewer | only if VRAM-bound on A100 |

Three facts to design around: CPU is effectively free and covers most of this project; **units burn while the runtime is connected, not while it computes**, making idle attached GPUs the largest waste vector; the competition inference runtime is an A100, so only final timing validation needs one.

Build:

- **`runtime.py`** — detect accelerator, expose `current_tier()` → `cpu|t4|l4|a100|h100|other`.
- **Tier declarations** — `@task(name=..., requires="cpu", max_tier="cpu")`.
- **The guard.** Below required tier → refuse, name the runtime to switch to. *Above* `max_tier` → **refuse by default**: `"features.structural is a CPU task but you're on an A100 (~12 units/hr). Switch to CPU, or pass allow_waste=True."`
- **`budget.py`** — ledger appending to `runs/budget.jsonl` (task, tier, wall seconds, estimated units, cumulative). `budget.report` shows spend by task and tier against a configured balance.
- **Idle watchdog** — after 20 min idle on a paid tier, warn loudly, optionally disconnect.
- **`shutdown_after=True`** on `run()` so an overnight job stops billing when it finishes.
- **`subsample=N`** on every expensive task. Nothing expensive runs full-scale before running green on a subsample.

**Cheap-first ladder, documented as default strategy:** CPU baselines → frozen-embedding + GBDT → fine-tuning only if the ladder is exhausted. Frozen embeddings are extracted **once** on L4, cached to Drive as Parquet, reused forever. Make the cache check unconditional and loud.

---

## 5. Progress visibility — every cell, every time

I need progress with an ETA on every cell, so I can judge whether to let a paid GPU keep running.

- **`tqdm.auto`** throughout, so it renders in both Colab and terminal.
- Every main loop wrapped with a descriptive `desc=`. Nested bars for folds: outer = folds, inner = batches. `leave=True` on top-level bars.
- Bars on file-walking, parsing, encoding, training, and inference — not just training.
- For long **non-iterative** operations (LightGBM fit, model load, zip extraction): a `heartbeat()` context manager printing elapsed time every N seconds, plus native callbacks where available (LightGBM `log_evaluation`, HF `TrainerCallback`).
- `run()` prints a header before starting: task, tier, subsample, cache status, and estimated duration if a prior run exists in `runs/`. On completion: wall time, units consumed, key metric, output path.
- **Critical: progress bars must be hard-disabled in submission mode.** The container caps logging at 500 lines × 500 chars and tqdm's carriage returns can blow through it. Gate every bar behind `PROGRESS_ENABLED`, which `main.py` sets to `False`. `submission.verify` checks this.

---

## 6. The Colab ⇄ GitHub loop

`notebooks/Trace_the_Ace_Runner.ipynb`. Must stay stable — all logic lives in the package. Execution is on Colab Pro+; GitHub is the source of truth; Colab is disposable. When something breaks: I fix it here with you, push, and re-run **one cell** — no runtime reset.

**Cell 1 — once per runtime.** Defines `sync()` and `run()` **inline in the notebook** (do not import these from a repo file — they would go stale and defeat the purpose):

```python
import os, sys, subprocess, traceback
from google.colab import drive

drive.mount("/content/drive")
REPO_URL = "https://github.com/meteorboyF/Trace-the-Ace--meteorboyF.git"
REPO_DIR = "/content/Trace-the-Ace"
DRIVE_ROOT = "/content/drive/MyDrive/trace-the-ace"
BRANCH = "main"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)


def sync(branch=BRANCH, install=False):
    """git pull -> purge cached modules -> re-import. Returns the package."""
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--quiet"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", branch, "--quiet"], check=True)
    subprocess.run(
        ["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{branch}", "--quiet"], check=True
    )
    src = os.path.join(REPO_DIR, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                os.path.join(REPO_DIR, "requirements-colab.txt"),
            ],
            check=True,
        )
    for m in [m for m in list(sys.modules) if m == "traceace" or m.startswith("traceace.")]:
        del sys.modules[m]
    import traceace

    traceace.configure(repo_dir=REPO_DIR, drive_root=DRIVE_ROOT)
    sha = subprocess.run(
        ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"synced @ {sha}")
    return traceace


def run(task, **kw):
    """sync, then run a task. Never kills the kernel on failure."""
    tc = sync()
    try:
        return tc.tasks.run(task, **kw)
    except Exception:
        traceback.print_exc()
        return None
```

- `reset --hard` is deliberate — local Colab edits are discarded.
- Module purge must catch the package and every submodule so a fresh import picks up new code without a restart.
- `install=True` is opt-in; pip installs persist for the runtime's life.
- `run()` swallows exceptions and prints a full traceback — a crash never costs the runtime.
- On `configure()`, print a status header: accelerator, tier, units/hr, session elapsed, cumulative units, staging status.

**Every subsequent cell is preceded by a markdown cell naming the required runtime:**

> ### 🖥️ RUNTIME: **CPU + High RAM** — ~0 units/hr · est. 10 min
> `run("features.structural")`
> Do **not** run on GPU.

> ### ⚡ RUNTIME: **L4 GPU** — ~5 units/hr · est. 40 min ≈ 3 units
> `run("features.embeddings", subsample=500)` ← smoke first
> `run("features.embeddings")`
> Cached to Drive. Should run **exactly once** per model+config.

Group cells by required runtime so I minimize switches. Include a session close-out cell that syncs artifacts, prints the budget report, and disconnects.

---

## 7. Self-verification — prove the code before I spend a unit

You have the real data locally. Use it. **Do not push until all five pass**, then report a pass/fail table.

1. **`pytest` green.**
2. **`ruff` and `mypy` clean.**
3. **`selftest.all`** — end-to-end on a tiny subsample (~200 responses / 50 sessions) using the **real local data**: ingest → consolidate → cv.build → all feature blocks → baseline → gbdt → calibrate → evaluate → interpret → submission.build → submission.verify. Must finish on CPU, on a laptop, in under five minutes, producing a valid `submission.csv`.
4. **`submission.smoke`** — run the built `main.py` against `submission_format_smoke.csv` and a matching 100-response sample. Assert it finishes well inside 10 minutes with correct format.
5. **Runtime parity check** — read `https://github.com/drivendataorg/tutoring-outcomes-runtime`, extract the pinned package list, reconcile against `requirements-colab.txt`. Anything we depend on that is absent from their image gets flagged in `docs/EXTERNAL_ASSETS.md` as needing a package-addition request via their GitHub issue process. **Do this early — a late discovery here is fatal.**

---

## 8. Storage — Drive is a FUSE mount

~100–300ms **per file operation** regardless of size. Sequential reads of one large file are fine; thousands of small reads are catastrophic. **Drive holds a handful of large files; `/content` local SSD is the working disk.**

```
/content/drive/MyDrive/trace-the-ace/
├── data/raw.zip
├── data/interim/transcripts.parquet   # zstd, consolidated
├── data/features/*.parquet
├── artifacts/models/*.safetensors     # individually large, direct write ok
├── artifacts/oof/*.parquet
├── artifacts/rollup.tar.zst           # everything else, batched
└── runs/
```

- **`staging.py`** with `stage_local()`: copy the single archive Drive → `/content/work`, extract **locally**, no-op if staged. Default on. Re-staging after a reset must cost under a minute.
- All reads/writes target `/content/work`. Only `sync_to_drive()` touches Drive, tarring first.
- **Never** `cp -r`, `shutil.copytree`, or glob iteration against a Drive path. A guard in `paths.py` raises if a Drive path reaches a file-iterating function.
- Extracting *to* Drive must be impossible by construction.
- Checkpoint helper syncs on an interval, so a disconnect costs at most one interval.

I will also upload the raw data to a Drive folder — support a `data.fetch_from_drive` path accepting a shared link, but prefer the uploaded `raw.zip`.

---

## 9. Repo structure

```
.
├── CLAUDE.md  README.md  pyproject.toml  requirements-colab.txt
├── .gitignore  .pre-commit-config.yaml
├── conf/{base.yaml, experiments/*.yaml}
├── docs/                             # §11
├── notebooks/Trace_the_Ace_Runner.ipynb
├── src/traceace/
│   ├── __init__.py                   # configure(), tasks
│   ├── config.py  paths.py  runtime.py  budget.py  staging.py
│   ├── io.py  cache.py  cv.py  progress.py
│   ├── features/{structural,linguistic,temporal,embeddings}.py
│   ├── models/{baseline,gbdt,encoder}.py
│   ├── calibration.py  evaluate.py  interpret.py  ensemble.py
│   ├── tasks.py  logging_utils.py
│   └── packaging/{build_submission,main_template,verify}.py
├── submission/
└── tests/
```

---

## 10. Tasks and modeling correctness

`tasks.py` — decorator registry, `run(name, **kwargs)`. Every task: tier-declared, idempotent (cache check, loud skip, `force=True`), checkpointed, manifest-logged (git SHA, config, wall time, tier, units, scores), subsample-capable, progress-instrumented.

| Task | Tier | Notes |
|---|---|---|
| `data.ingest` | cpu | normalize suffixed filenames, validate schemas |
| `data.consolidate` | cpu | transcripts → single Parquet, once |
| `eda.overview` | cpu | shapes, label balance, responses-per-session, missingness |
| `eda.transcripts` | cpu | **exact** token/char distributions, role balance, timing gaps |
| `eda.inference_budget` | cpu | go/no-go table vs the 6h cap — §3 |
| `cv.build` | cpu | folds generated **once**, persisted, reused |
| `baseline.prior` | cpu | global base rate — the floor |
| `baseline.lo_only` | cpu | **the bar every real model must clear** |
| `features.structural` | cpu | turn counts, talk ratio, utterance length stats |
| `features.linguistic` | cpu | question density, hedging/confusion markers, affirmations, student answer-length trajectory |
| `features.temporal` | cpu | inter-utterance latency, response-time trends |
| `features.embeddings` | l4 | **cached forever, once per model+config** |
| `model.gbdt` | cpu | LightGBM on CPU beats GPU at this data size |
| `model.encoder` | l4/a100 | only after the cheap ladder |
| `calibrate.fit` | cpu | isotonic + Platt on OOF |
| `ensemble.blend` | cpu | log-loss-optimal weights over OOF |
| `evaluate.report` | cpu | log loss, AUC, reliability curve, per-slice |
| `interpret.report` | cpu | §11 |
| `submission.build` / `.verify` / `.smoke` | cpu / cpu / a100 | |
| `selftest.all` / `budget.report` / `docs.build` | cpu | |

**Do not get these wrong:**

1. **Fold splitting groups by `session_id`, never `response_id`.** One session produces multiple response rows (multiple learning objectives) — splitting by response leaks the transcript across folds and every score becomes fiction. Use `StratifiedGroupKFold(groups=session_id, y=correct)`. Write a test asserting zero session overlap, run it in CI.
2. **Always persist OOF predictions** to `artifacts/oof/`, keyed by experiment. Calibration and blending depend on them, and log loss is very responsive to calibration.
3. **Log loss is the objective.** Headline it everywhere. Clip predictions away from exact 0/1.
4. **Every report shows the delta vs `baseline.lo_only`** — the organizers' stated anti-goal made structurally impossible to ignore.
5. Seed everything; record the seed in the manifest.
6. No test-set transduction, per §2.

---

## 11. Research artifacts and documentation

I am targeting the **publication bonus** (write-ups developed into publishable papers), so research output is a primary deliverable, not a byproduct.

`interpret.report` runs alongside every model task, producing into `artifacts/figures/` and `docs/FINDINGS.md`:

- Cross-fold feature importance with confidence intervals, not a single-fit bar chart.
- Per-slice performance by transcript length, turn count, learning-objective family, student talk ratio — where does it work, where does it fail?
- Reliability diagrams before and after calibration.
- **Key-moments attribution over transcript position** — which segments of a long session carry signal? Early diagnostic turns, mid-session struggle, closing checks for understanding?
- **Tutoring-move taxonomy** — cluster or classify tutor utterances into move types (questioning, explaining, scaffolding, affirming, correcting) and relate move distribution to outcome.
- Ablation harness giving each feature block's marginal contribution, so claims are about *what* mattered, not *that* something worked.

Figures publication-quality by default: labelled axes, readable at print size, PNG + PDF, colourblind-safe palette. The write-up is 4 pages including figures — plan for direct reuse.

**`docs/FINDINGS.md` is a living paper draft from day one**, not a notes file:

```
Abstract (placeholder, filled last)
Key findings          — numbered claims, each with supporting evidence + figure
Methodology           — feature engineering, modeling, validation, interpretability
Extensions & generalizability — transfer to other chat-based tutoring setups,
                        limitations, when to distrust outputs, future work
References
```

Every claim carries a pointer to the run manifest and figure backing it. Write for education researchers who are not ML experts — no unexplained jargon. Add to it after every `interpret.report`, not at the end. **Log negative results too** — failed approaches are publishable findings and the Rigor criterion rewards them. Keep a running page count against the 4-page limit.

Other docs, kept current:

- **`STATE.md`** — read-this-first. Status, what works, best score and which experiment, known problems, next actions, units remaining. Updated at the end of every session unprompted. "Last updated / git SHA" at top.
- **`ARCHITECTURE.md`** — how the pieces fit and *why*.
- **`RUNBOOK.md`** — starting a session, runtime per task, recovering from a disconnect, adding a task, building/verifying a submission, what to do when a cell errors.
- **`DECISIONS.md`** — append-only dated ADR log: context, decision, alternatives rejected, consequences.
- **`EXPERIMENTS.md`** — auto-generated from `runs/` by `docs.build`: date, SHA, config, CV log loss, LB score, units, notes, sorted by score.
- **`DATA.md`** — measured facts: row counts, session counts, token distributions, label balance.
- **`EXTERNAL_ASSETS.md`** — every external model/dataset with license, URL, commercial-use verdict, runtime availability.
- **`COMPETITION.md`** — the §2 constraints condensed to what affects engineering decisions.

Docstrings carrying reasoning, type hints throughout, comments explaining *why*.

---

## 12. Submission packaging and hardening

`submission.build` → `submission.zip`, `main.py` at **root level**, assets alongside. Constraints in §2.

`submission.verify` must fail loudly on: `main.py` not at zip root; row set or ordering mismatch vs `submission_format.csv`; any probability outside `[0,1]`, NaN, or missing `response_id`; **any print/log emitting test data** — statically scan `main.py` and its imports, reject prints in data-handling paths; progress bars not disabled; any import that could touch the network; projected runtime above 4.5h against the 6h cap; zip above 55GB; total log lines above 400.

`tests/`: CV no-leakage assertion, submission format round-trip, cache invalidation, tier-guard behaviour, staging idempotency, Drive-iteration path guard, filename normalization, progress-disabled-in-submission-mode. GitHub Actions running lint + tests on push with synthetic fixtures (no data). `ruff` + `mypy` in `pyproject.toml`. Fail fast and loudly on bad config, never silently default. Deterministic seeding at every entry point. Structured logs to `artifacts/logs/` with rotation.

**License allowlist** for `CLAUDE.md`: any external model or dataset must permit commercial use — no NC, CC BY-NC, or research-only terms. Apache-2.0 and MIT are safe (Qwen, Mistral, ModernBERT, DeBERTa). Flag bespoke community licenses as needing a forum question before I build on them.

---

## 13. Build order

1. `.gitignore` + pre-commit hook + `git status`. **Report and pause.**
2. `docs/BRIEF.md`, `CLAUDE.md`, environment verification. **Report.**
3. Skeleton: `pyproject.toml`, `requirements-colab.txt`, `README.md`.
4. `config.py`, `paths.py` (Drive guard), `logging_utils.py`, `progress.py`.
5. `runtime.py`, `budget.py`, `staging.py`.
6. `tasks.py` registry with tier guard + `configure()`.
7. `data.ingest`, `io.py`, `cache.py`, `data.consolidate` — **run against the real local data, report actual counts.**
8. `eda.overview`, `eda.transcripts`, `eda.inference_budget` — **run them, report the numbers, state your architecture recommendation with the arithmetic shown. Pause here for my input.**
9. `cv.py` + leakage test.
10. `baseline.*`, `evaluate.report`.
11. Feature modules with real caching and progress bars.
12. `interpret.py`, `packaging/`.
13. `docs/` populated for real, not placeholders.
14. Notebook with runtime banners.
15. Tests + CI. Run the §7 suite. **Report the pass/fail table.** Then push.

---

## 14. Deliverables

- Tree of what you created.
- **Measured data facts** and the `eda.inference_budget` verdict, arithmetic shown.
- The §7 verification pass/fail table.
- Exact first three Colab commands and the runtime for each.
- Checklist of what I must place in Drive by hand, and where.
- Recommended first week of tasks in order, with runtime and estimated unit cost.
- Anything you deliberately deviated from, and why.

Prefer sensible defaults over blocking questions, but ask about anything genuinely ambiguous before building.
