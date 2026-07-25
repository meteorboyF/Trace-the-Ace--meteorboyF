"""Tier guard, Drive-path guard, cache invalidation, staging idempotency, progress switch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import traceace
from traceace import progress
from traceace.cache import feature_cache_path, load_or_compute
from traceace.paths import DrivePathError, assert_not_drive, is_drive_path, iter_files
from traceace.runtime import tier_at_least, tier_at_most, tier_rank


# --- tier guard --------------------------------------------------------------
def test_tier_ordering():
    assert tier_rank("cpu") < tier_rank("t4") < tier_rank("l4") < tier_rank("h100")
    assert tier_at_least("a100", "cpu")
    assert not tier_at_least("cpu", "l4")
    assert tier_at_most("cpu", "cpu")
    assert not tier_at_most("a100", "cpu")


def test_guard_refuses_when_below_required_tier(synth_repo, monkeypatch):
    """A task needing an L4 must refuse on CPU and name the runtime to switch to."""
    from traceace import runtime, tasks

    monkeypatch.setattr(runtime, "detect_accelerator",
                        lambda: runtime.Accelerator("cpu", "CPU", None))
    monkeypatch.setattr(tasks, "detect_accelerator",
                        lambda: runtime.Accelerator("cpu", "CPU", None))
    with pytest.raises(RuntimeError, match="needs at least 'l4'"):
        tasks.run("features.embeddings")


def test_guard_refuses_wasteful_tier_by_default(synth_repo, monkeypatch):
    """A CPU task must refuse to run on an A100 unless allow_waste=True."""
    from traceace import runtime, tasks

    fake = runtime.Accelerator("a100", "NVIDIA A100-SXM4-80GB", 81920)
    monkeypatch.setattr(tasks, "detect_accelerator", lambda: fake)
    with pytest.raises(RuntimeError, match="wastes units"):
        tasks.run("eda.overview")


# --- Drive path guard --------------------------------------------------------
def test_drive_path_detection(tmp_path):
    drive = tmp_path / "drive"
    (drive / "data").mkdir(parents=True)
    traceace.configure(repo_dir=_repo_root(), drive_root=drive, work_dir=tmp_path / "work",
                       quiet=True)
    assert is_drive_path(drive / "data" / "x.parquet")
    assert not is_drive_path(tmp_path / "work" / "x.parquet")


def test_iterating_drive_path_raises(tmp_path):
    drive = tmp_path / "drive"
    (drive / "data").mkdir(parents=True)
    traceace.configure(repo_dir=_repo_root(), drive_root=drive, work_dir=tmp_path / "work",
                       quiet=True)
    with pytest.raises(DrivePathError):
        list(iter_files(drive / "data", "*.csv"))
    with pytest.raises(DrivePathError):
        assert_not_drive(drive / "data")


def test_local_path_iteration_allowed(tmp_path):
    work = tmp_path / "work"
    (work / "d").mkdir(parents=True)
    (work / "d" / "a.csv").write_text("x")
    traceace.configure(repo_dir=_repo_root(), drive_root=tmp_path / "drive", work_dir=work,
                       quiet=True)
    assert [p.name for p in iter_files(work / "d", "*.csv")] == ["a.csv"]


# --- cache -------------------------------------------------------------------
def test_cache_hit_skips_recompute(synth_repo):
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    path = feature_cache_path("unit_test_block", "v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    load_or_compute(path, compute)
    load_or_compute(path, compute)
    assert calls["n"] == 1, "second call must hit the cache"

    load_or_compute(path, compute, force=True)
    assert calls["n"] == 2, "force=True must recompute"


def test_cache_version_invalidates(synth_repo):
    assert feature_cache_path("blk", "v1") != feature_cache_path("blk", "v2")


# --- staging idempotency -----------------------------------------------------
def test_stage_local_is_noop_when_already_staged(tmp_path, monkeypatch):
    work = tmp_path / "work"
    tdir = work / "data" / "raw" / "train_transcripts"
    tdir.mkdir(parents=True)
    (tdir / "s1.csv").write_text("session_id,utterance_id,role,content,timestamp\n")
    traceace.configure(repo_dir=_repo_root(), drive_root=None, work_dir=work, quiet=True)

    from traceace import staging

    assert staging.is_staged()
    calls = {"extract": 0}
    monkeypatch.setattr(staging, "_extract_archive",
                        lambda *a, **k: calls.__setitem__("extract", calls["extract"] + 1))
    staging.stage_local()
    staging.stage_local()
    assert calls["extract"] == 0, "staging must be a no-op when already staged"


# --- progress kill switch ----------------------------------------------------
def test_progress_disabled_by_env(monkeypatch):
    monkeypatch.setenv("TRACEACE_PROGRESS", "0")
    assert progress.progress_enabled() is False


def test_progress_bar_disabled_produces_no_output(monkeypatch, capsys):
    monkeypatch.setenv("TRACEACE_PROGRESS", "0")
    for _ in progress.pbar(range(5), desc="should be silent"):
        pass
    captured = capsys.readouterr()
    assert "should be silent" not in captured.out + captured.err


def test_heartbeat_silent_when_disabled(monkeypatch, capsys):
    monkeypatch.setenv("TRACEACE_PROGRESS", "0")
    with progress.heartbeat("quiet", every_seconds=0.01):
        pass
    captured = capsys.readouterr()
    assert "quiet" not in captured.out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# --- OOF namespacing: regression test for a real bug --------------------------
def test_subsampled_oof_cannot_clobber_full_oof(synth_repo):
    """A smoke run must never overwrite a full-data experiment's OOF.

    Regression test. Originally OOF was keyed by experiment name alone, so a
    400-session selftest overwrote the full-data `baseline.lo_only` OOF and every
    subsequent `delta_vs_lo_only` was silently measured against the wrong baseline —
    the headline number flipped sign with no error raised.
    """
    import pandas as pd

    from traceace.evaluate import experiment_name, load_oof, oof_path, save_oof
    from traceace.io import LABEL_COL

    full = pd.DataFrame({
        "response_id": [f"r{i}" for i in range(10)],
        "session_id": [f"s{i}" for i in range(10)],
        LABEL_COL: [1.0, 0.0] * 5,
        "pred": [0.9] * 10,
    })
    sub = full.head(4).copy()
    sub["pred"] = 0.1

    save_oof("baseline.lo_only", full)                    # full-data run
    save_oof("baseline.lo_only", sub, subsample=400)      # smoke run

    assert oof_path(experiment_name("baseline.lo_only", None)) != oof_path(
        experiment_name("baseline.lo_only", 400)
    )
    # the full-data OOF must be untouched
    reloaded = load_oof("baseline.lo_only")
    assert len(reloaded) == 10
    assert float(reloaded["pred"].iloc[0]) == 0.9
    assert len(load_oof("baseline.lo_only", subsample=400)) == 4


def test_baseline_logloss_is_subsample_matched(synth_repo):
    """delta_vs_lo_only must compare like-for-like row sets."""
    import pandas as pd

    from traceace.evaluate import baseline_logloss, save_oof
    from traceace.io import LABEL_COL

    full = pd.DataFrame({
        "response_id": [f"r{i}" for i in range(20)],
        "session_id": [f"s{i}" for i in range(20)],
        LABEL_COL: [1.0, 0.0] * 10,
        "pred": [0.5] * 20,
    })
    sub = full.head(6).copy()
    sub["pred"] = 0.8
    save_oof("baseline.lo_only", full)
    save_oof("baseline.lo_only", sub, subsample=400)

    assert baseline_logloss() != baseline_logloss(subsample=400)


def test_model_artifact_dirs_are_subsample_namespaced(synth_repo):
    """A smoke run must not overwrite the fold models a submission is built from.

    Regression test for the more dangerous sibling of the OOF bug: `submission.build`
    reads `artifacts/models/<experiment>/fold*.txt`. Unnamespaced, a 400-session selftest
    overwrote those, and the submission would have shipped a model trained on 1.7% of the
    data — silently, because the files are valid LightGBM boosters either way.
    """
    from traceace.evaluate import experiment_dir

    full = experiment_dir("model.gbdt", None)
    sub = experiment_dir("model.gbdt", 400)
    assert full != sub
    assert full.name == "model_gbdt"
    assert "sub400" in sub.name


def test_submission_build_reads_full_data_models_only():
    """submission.build must pin to the full-data dir regardless of any subsample kwarg."""
    import inspect

    from traceace.packaging import build_submission

    src = inspect.getsource(build_submission)
    # every artifact lookup in the builder passes an explicit None (full data)
    assert "experiment_dir(experiment, None)" in src
    assert "models_dir() / experiment.replace" not in src
