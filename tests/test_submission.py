"""Submission hardening: format round-trip, the static leak scanner, filename normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traceace.evaluate import clip_probs, logloss
from traceace.packaging.verify import (
    VerifyResult,
    check_progress_disabled,
    scan_source,
    verify_predictions,
)


# --- static scan: the disqualification-risk check ----------------------------
def test_scan_flags_print_of_variable():
    src = "def f(df):\n    print(len(df))\n"
    leaks, _ = scan_source(src, "main.py")
    assert leaks, "printing a computed value must be flagged"


def test_scan_flags_fstring_with_interpolation():
    src = 'def f(n):\n    print(f"rows={n}")\n'
    leaks, _ = scan_source(src, "main.py")
    assert leaks, "f-string interpolation could leak test data"


def test_scan_allows_static_string():
    src = 'def f():\n    print("start")\n'
    leaks, _ = scan_source(src, "main.py")
    assert not leaks


def test_scan_allows_sanctioned_log_wrapper():
    """print(msg) inside `def log(msg)` is the approved static-logging funnel."""
    src = 'def log(msg: str) -> None:\n    print(msg, flush=True)\n\nlog("start")\n'
    leaks, _ = scan_source(src, "main.py")
    assert not leaks


def test_scan_flags_leak_through_log_call_site():
    src = (
        'def log(msg: str) -> None:\n    print(msg, flush=True)\n\n'
        'def g(df):\n    log(f"n={len(df)}")\n'
    )
    leaks, _ = scan_source(src, "main.py")
    assert leaks, "a non-literal at the log() CALL SITE must still be caught"


def test_scan_flags_network_imports():
    for mod in ("requests", "huggingface_hub", "socket"):
        _, net = scan_source(f"import {mod}\n", "main.py")
        assert net, f"{mod} should be flagged as network-capable"


def test_scan_allows_offline_imports():
    _, net = scan_source("import numpy\nimport pandas\nimport lightgbm\n", "main.py")
    assert not net


def test_progress_disabled_detection():
    assert check_progress_disabled('os.environ["TRACEACE_PROGRESS"] = "0"')
    assert check_progress_disabled('os.environ["TQDM_DISABLE"] = "1"')
    assert not check_progress_disabled("import pandas as pd")


# --- submission format round-trip -------------------------------------------
def test_verify_predictions_round_trip(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(20)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format",
                        lambda smoke=False: fmt)

    good = tmp_path / "good.csv"
    pd.DataFrame({"response_id": fmt["response_id"],
                  "probability": np.linspace(0.01, 0.99, 20)}).to_csv(good, index=False)
    res = VerifyResult()
    verify_predictions(good, res)
    assert res.ok, [c.name for c in res.failures]


def test_verify_catches_wrong_row_order(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(20)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format",
                        lambda smoke=False: fmt)

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"response_id": fmt["response_id"][::-1].to_numpy(),
                  "probability": 0.5}).to_csv(bad, index=False)
    res = VerifyResult()
    verify_predictions(bad, res)
    assert any(c.name == "row_ORDER_matches" and not c.passed for c in res.checks)


def test_verify_catches_out_of_range_probability(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(5)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format",
                        lambda smoke=False: fmt)

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"response_id": fmt["response_id"],
                  "probability": [0.5, 1.5, -0.2, 0.3, 0.4]}).to_csv(bad, index=False)
    res = VerifyResult()
    verify_predictions(bad, res)
    assert any(c.name == "probabilities_in_unit_interval" and not c.passed
               for c in res.checks)


def test_verify_catches_nan(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(4)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format",
                        lambda smoke=False: fmt)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"response_id": fmt["response_id"],
                  "probability": [0.5, np.nan, 0.3, 0.4]}).to_csv(bad, index=False)
    res = VerifyResult()
    verify_predictions(bad, res)
    assert any(c.name == "no_nan_probabilities" and not c.passed for c in res.checks)


# --- metric sanity -----------------------------------------------------------
def test_clip_prevents_infinite_logloss(synth_repo):
    y = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])  # maximally wrong and unclipped -> inf
    assert np.isfinite(logloss(y, p))


def test_clip_probs_bounds(synth_repo):
    p = clip_probs(np.array([0.0, 1.0, 0.5]))
    assert (p > 0).all() and (p < 1).all()


# --- filename normalization --------------------------------------------------
def test_submission_formats_distinguished_by_row_count(tmp_path, synth_repo, monkeypatch):
    """The two format files must be told apart by ROW COUNT, never by filename."""
    from traceace import data
    from traceace.paths import raw_dir

    rdir = raw_dir()
    rdir.mkdir(parents=True, exist_ok=True)
    big = pd.DataFrame({"response_id": [f"r{i}" for i in range(500)], "probability": 0.5})
    small = pd.DataFrame({"response_id": [f"s{i}" for i in range(10)], "probability": 0.5})
    # deliberately misleading names: the "small" suffix sorts first
    big.to_csv(rdir / "submission_format_AAA.csv", index=False)
    small.to_csv(rdir / "submission_format_ZZZ.csv", index=False)

    data.ingest(force=True)
    assert len(pd.read_csv(rdir / "submission_format.csv")) == 500
    assert len(pd.read_csv(rdir / "submission_format_smoke.csv")) == 10


def test_classify_csv_by_content_shape(tmp_path):
    from traceace.data import _classify_csv

    f = tmp_path / "weird_name_123.csv"
    f.write_text("response_id,session_id,learning_objective_id,learning_objective\na,b,c,d\n")
    assert _classify_csv(f) == "train_features"

    g = tmp_path / "other.csv"
    g.write_text("response_id,is_correct\na,1.0\n")
    assert _classify_csv(g) == "train_labels"
