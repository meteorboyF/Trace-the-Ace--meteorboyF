# RUNBOOK.md — operational recipes

## Starting a session (every time)

1. Read [`STATE.md`](STATE.md) — status, best score, next actions, units left.
2. Open `notebooks/Trace_the_Ace_Runner.ipynb` in Colab.
3. **Pick the runtime named in the markdown banner above the cell you intend to run.**
   Default to **CPU + High RAM** — it is free and covers most of this project.
4. Run **Cell 1** (mounts Drive, clones/pulls, defines `sync()` and `run()`). It prints a
   status header: accelerator, tier, units/hr, session elapsed, cumulative units, staging.
5. Run the task cells you need.
6. Run the **close-out cell**: syncs artifacts to Drive, prints the budget report,
   disconnects.

> **The golden loop.** When something breaks: fix it *here* with Claude, `git push`, then
> re-run **one cell** in Colab (`run()` calls `sync()` first). No runtime reset, no
> re-staging, no lost GPU time. `run()` swallows exceptions and prints the traceback, so a
> crash never costs you the runtime.

## Which runtime for which task

| Runtime | units/hr | Tasks |
|---|---|---|
| **CPU + High RAM** | ~0 | `data.*`, `eda.*`, `cv.build`, `baseline.*`, `features.structural/linguistic/temporal/lo_alignment`, `model.gbdt`, `model.move_classifier`, `calibrate.*`, `ensemble.*`, `evaluate.*`, `interpret.*`, `submission.build/verify`, `selftest.all`, `docs.build`, `budget.report` |
| **L4** | ~5 | `features.embeddings` (once), `annotate.moves` (backend="vllm") |
| **A100** | ~12 | final `submission.smoke` timing validation only |

If you run a CPU task on a GPU, the tier guard **refuses** and tells you what to switch to.
Override only with `allow_waste=True`, and only deliberately.

## Local development (CPU, no GPU needed)

```bash
export PATH="$HOME/.local/bin:$PATH"          # uv
.venv/bin/python -c "import traceace; traceace.configure(repo_dir='.'); \
    traceace.tasks.run('eda.overview')"

# Full pipeline end-to-end on real data, ~35s:
.venv/bin/python -c "import traceace; traceace.configure(repo_dir='.', quiet=True); \
    traceace.tasks.run('selftest.all')"

# Quality gates (all must be green before pushing):
.venv/bin/ruff check . && .venv/bin/ruff format --check . \
  && .venv/bin/mypy && .venv/bin/pytest -q
```

The venv is uv-managed **Python 3.12.13**, matching the competition runtime exactly.
Recreate it with:
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

## Recovering from a Colab disconnect

1. Reconnect, choose the same runtime tier.
2. Run **Cell 1** only. It re-clones/pulls and re-`configure()`s.
3. `stage_local()` is a **no-op if already staged**; a fresh VM re-stages from the single
   `raw.zip` on Drive in under a minute (one big sequential copy, then local extraction).
4. Cached feature blocks under `data/features/` are reused automatically — the cache check
   is loud, so you will see `CACHE HIT` in the log. Nothing recomputes.
5. Check `budget.report` to see what the disconnect cost.

## Adding a new task

1. Write the function in the appropriate module; decorate it:
   ```python
   @task("features.myblock", requires="cpu", max_tier="cpu", description="one-line summary")
   def build(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
       ...
       return {"output_path": str(path), "n_rows": len(df)}  # dict => manifest metrics
   ```
2. Accept `force` and `subsample`; use `cache.load_or_compute` so reruns are free.
3. Wrap the main loop in `progress.pbar(..., desc="…")`; use `heartbeat()` for long
   non-iterative calls.
4. Add the module to `_TASK_MODULES` in `src/traceace/tasks.py` if it is new.
5. Return a dict — `logloss` / `delta_vs_lo_only` / `output_path` get headlined in the
   summary line.
6. If it is a feature block, register it in `features/assemble.py::BLOCKS` so ablation sees
   it.

## Building and verifying a submission

```python
run("model.gbdt")  # trains folds, writes OOF + importance
run("calibrate.fit", experiment="model.gbdt")
run("submission.build", experiment="model.gbdt")
run("submission.smoke")  # executes main.py as a subprocess
run("submission.verify", smoke=True)  # raises on any violation
```

`submission.verify` fails loudly on: `main.py` not at zip root · row set/ordering mismatch ·
probability outside [0,1] or NaN · non-literal print/log (AST scan) · progress bars enabled ·
network-capable imports · projected runtime > 4.5 h · zip > 55 GB · > 400 log lines.

**Only three full submissions per week** (~14 attempts left). Smoke tests, cancelled and
failed jobs do not count.

## Submitting for real — the exact sequence

**Step 0 — local (already automated).**
```bash
.venv/bin/python -c "import traceace; traceace.configure(repo_dir='.', quiet=True); \
    traceace.tasks.run('selftest.all')"
```
Builds, smoke-runs `main.py` as a subprocess, and runs all 18 verify checks. Must be green.
The built artifact is `submission/submission.zip`.

**Step 1 — the organizers' local harness.** They ask for this before submitting:
```bash
git clone https://github.com/drivendataorg/tutoring-outcomes-runtime
cd tutoring-outcomes-runtime
# place OUR zip where their harness expects it
cp /home/meteorboyf/Datathons/TraceTheAce/submission/submission.zip submission/submission.zip
# their harness needs a data/ dir shaped like the container; follow their README
just test-submission
```
This runs our `main.py` inside their actual container image, which catches anything that
depends on our local Python environment. If `just` is not installed: `sudo apt install just`
or use the equivalent `make`/docker command from their README.

**Step 2 — smoke environment on the platform.** Upload `submission.zip` and choose the
**smoke** option. Smoke runs do **not** count against the 3/week limit. Must finish inside
10 minutes; ours projects ~5 s for 100 rows.

**Step 3 — full submission.** Only after the smoke run succeeds. This **does** consume one of
the three weekly slots.

**What to check on the leaderboard.** Our CV says 0.54339 (safe variant). If the LB score is
far worse than that, suspect train/serve skew first — that is exactly what a real submission
is for, and why submitting early is worth a slot even with a modest model.

## When a cell errors

- `run()` never kills the kernel — read the printed traceback.
- **`RuntimeError: [tier-guard] …`** → you are on the wrong runtime. Switch, or pass
  `allow_waste=True` if you truly mean it.
- **`FileNotFoundError: … folds.parquet`** → run `cv.build` first.
- **`FileNotFoundError: feature block … not built`** → run that `features.*` task.
- **`traceace is not configured`** → run Cell 1.
- **`DrivePathError`** → something tried to iterate the Drive mount. Stage locally first;
  never point a file-walking function at Drive.
- **Constant predictions in the smoke output** → features are not reaching the model
  (train/serve skew). `selftest.all` asserts against this; check
  `tests/test_inference_parity.py`.
- Fix locally → `git push` → re-run the single cell. Do **not** edit code in Colab; `sync()`
  does `reset --hard` and your edits will be discarded.

## Cost hygiene

- Never leave a GPU runtime attached while thinking — units burn on **connection**, not
  computation.
- Smoke every expensive task first: `run("features.embeddings", subsample=500)`.
- Use `shutdown_after=True` for long unattended jobs so billing stops on completion.
- `run("budget.report")` shows spend by task and tier against the 733-unit balance.
- Frozen embeddings are extracted **once per model+config** and cached forever. If you see
  extraction start a second time, stop and investigate the cache key.
