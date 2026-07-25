"""End-to-end self-verification on the real local data (§7).

``selftest.all`` runs the entire pipeline on a tiny subsample — ingest → consolidate →
cv.build → every feature block → baselines → gbdt → calibrate → evaluate → interpret →
submission.build → submission.verify — on **CPU, on a laptop, in under five minutes**,
and asserts a valid ``submission.csv`` comes out the far end.

``submission.smoke`` mimics the competition smoke environment: it builds a temporary
``data/`` directory shaped exactly like the container's (read-only ``submission_format.csv``,
``test_features.csv``, ``test_transcripts/``) from a 100-response training sample, then
executes the packaged ``main.py`` as a **subprocess** — the same way the organizers will —
and asserts it finishes well inside 10 minutes with a correctly formatted output.

Running main.py as a subprocess (rather than importing it) is deliberate: it catches
import-time failures, path assumptions, and stray stdout that an in-process call would
mask.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_config
from .io import load_submission_format, load_train_features
from .logging_utils import get_logger
from .paths import submission_dir, transcripts_dir
from .tasks import run as run_task
from .tasks import task

log = get_logger("selftest")


@task(
    "selftest.all",
    requires="cpu",
    max_tier="cpu",
    description="end-to-end pipeline on a tiny real-data subsample (<5 min, CPU)",
)
def all_(
    force: bool = True,
    subsample: int | None = None,
    n_sessions: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    n_sessions = int(n_sessions or cfg.get("selftest", "n_sessions", default=50))
    max_seconds = float(cfg.get("selftest", "max_seconds", default=300))
    t0 = time.monotonic()

    steps: list[dict[str, Any]] = []

    def step(name: str, **kw: Any) -> Any:
        s = time.monotonic()
        res = run_task(name, **kw)
        steps.append(
            {
                "task": name,
                "seconds": round(time.monotonic() - s, 2),
                "ok": True,
            }
        )
        return res

    step("data.ingest", force=False)
    step("data.consolidate", subsample=n_sessions, force=force)
    step("cv.build", subsample=n_sessions, force=force)

    step("features.structural", subsample=n_sessions, force=force)
    step("features.linguistic", subsample=n_sessions, force=force)
    step("features.temporal", subsample=n_sessions, force=force)
    step("features.lo_alignment", subsample=n_sessions, force=force)

    step("baseline.prior", subsample=n_sessions)
    step("baseline.lo_only", subsample=n_sessions)

    # min_data_in_leaf must be small here or the subsample is too tiny to split at all
    # and the model degenerates to a constant — which would hide real bugs.
    gbdt = step(
        "model.gbdt",
        subsample=n_sessions,
        num_boost_round=120,
        early_stopping_rounds=30,
        params={"min_data_in_leaf": 5, "num_leaves": 15},
    )
    step("calibrate.fit", experiment="model.gbdt", subsample=n_sessions)
    step("evaluate.report", experiment="model.gbdt")
    step("interpret.report", experiment="model.gbdt", subsample=n_sessions)

    step("annotate.moves", subsample=n_sessions, n_sample=2000, force=force)
    step("model.move_classifier", subsample=None, force=force)

    built = step("submission.build", experiment="model.gbdt")
    smoke = step("submission.smoke")
    step(
        "submission.verify",
        submission_csv=smoke.get("submission_csv") if smoke else None,
        smoke=True,
        strict=True,
    )

    # A constant prediction vector passes every format check but means the feature
    # pipeline reached the model as all-NaN (train/serve skew) — assert it varies.
    n_unique = _n_unique_predictions(smoke)
    if n_unique <= 1:
        raise AssertionError(
            "submission.csv predictions are CONSTANT — features are not reaching the "
            "model at inference (likely train/serve skew). Investigate before pushing."
        )

    elapsed = time.monotonic() - t0
    res = {
        "steps": steps,
        "total_seconds": round(elapsed, 1),
        "within_budget": elapsed <= max_seconds,
        "budget_seconds": max_seconds,
        "gbdt_logloss": gbdt.get("logloss") if gbdt else None,
        "submission_zip": built.get("output_path") if built else None,
        "n_unique_predictions": n_unique,
        "n_sessions": n_sessions,
    }
    if not res["within_budget"]:
        log.warning("selftest.all took %.1fs (budget %.0fs)", elapsed, max_seconds)
    log.info("selftest.all completed in %.1fs", elapsed)
    return res


def _n_unique_predictions(smoke_result: dict[str, Any] | None) -> int:
    """Number of distinct probabilities in the smoke submission (1 == degenerate)."""
    if not smoke_result:
        return 0
    path = Path(smoke_result.get("submission_csv", ""))
    if not path.is_file():
        return 0
    return int(pd.read_csv(path)["probability"].round(9).nunique())


@task(
    "submission.smoke",
    requires="cpu",
    max_tier="a100",
    description="run the packaged main.py against a 100-response smoke set, as a subprocess",
)
def smoke(
    force: bool = False,
    subsample: int | None = None,
    zip_name: str = "submission.zip",
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Mimic the competition smoke environment and execute main.py exactly as they will."""
    sdir = submission_dir()
    zpath = sdir / zip_name
    if not zpath.is_file():
        raise FileNotFoundError(f"{zpath} missing — run submission.build first")

    workdir = sdir / "_smoke"
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "data").mkdir(parents=True, exist_ok=True)

    # unzip exactly as the organizers do: contents land in the working dir root
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(workdir)
    if not (workdir / "main.py").is_file():
        raise AssertionError("main.py is not at the zip root — the submission would fail")

    # --- build a container-shaped data/ dir from the smoke format -----------
    fmt = load_submission_format(smoke=True)
    feats = load_train_features()
    sample = feats[feats["response_id"].isin(set(fmt["response_id"]))]
    if sample.empty:
        # smoke ids may not overlap train; fall back to a same-sized training slice
        sample = feats.head(len(fmt)).copy()
        fmt = pd.DataFrame({"response_id": sample["response_id"].to_numpy(), "probability": 0.5})

    (workdir / "data" / "test_transcripts").mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for sid in sample["session_id"].unique():
        src = transcripts_dir() / f"{sid}.csv"
        if src.is_file():
            shutil.copyfile(src, workdir / "data" / "test_transcripts" / f"{sid}.csv")
            n_copied += 1
    sample.to_csv(workdir / "data" / "test_features.csv", index=False)
    fmt.to_csv(workdir / "data" / "submission_format.csv", index=False)

    # --- execute main.py as a subprocess -------------------------------------
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed = time.monotonic() - t0

    stdout_lines = proc.stdout.splitlines()
    stderr_tail = proc.stderr.splitlines()[-15:]
    if proc.returncode != 0:
        raise RuntimeError(
            f"main.py exited {proc.returncode} after {elapsed:.1f}s.\n"
            f"stderr tail:\n" + "\n".join(stderr_tail)
        )

    out_csv = workdir / "submission.csv"
    if not out_csv.is_file():
        raise AssertionError("main.py did not write submission.csv beside itself")

    got = pd.read_csv(out_csv, dtype={"response_id": str})
    n_test_full = len(load_submission_format(smoke=False))
    projected_hours = (elapsed / max(len(fmt), 1)) * n_test_full / 3600.0

    res = {
        "submission_csv": str(out_csv),
        "seconds": round(elapsed, 2),
        "within_10min": elapsed < 600.0,
        "n_rows": int(len(got)),
        "n_expected": int(len(fmt)),
        "rows_match": len(got) == len(fmt),
        "n_transcripts_copied": n_copied,
        "stdout_line_count": len(stdout_lines),
        "log_lines_under_400": len(stdout_lines) <= 400,
        "projected_full_hours": round(projected_hours, 3),
        "projected_under_4.5h": projected_hours <= 4.5,
        "workdir": str(workdir),
    }
    log.info(
        "submission.smoke: %.1fs, %d rows, projected %.2fh for full test",
        elapsed,
        len(got),
        projected_hours,
    )
    (sdir / "smoke_report.json").write_text(json.dumps(res, indent=2))
    return res
