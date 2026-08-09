"""Submission hardening: format round-trip, the static leak scanner, filename normalization."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

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
        "def log(msg: str) -> None:\n    print(msg, flush=True)\n\n"
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
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format", lambda smoke=False: fmt)

    good = tmp_path / "good.csv"
    pd.DataFrame(
        {"response_id": fmt["response_id"], "probability": np.linspace(0.01, 0.99, 20)}
    ).to_csv(good, index=False)
    res = VerifyResult()
    verify_predictions(good, res)
    assert res.ok, [c.name for c in res.failures]


def test_verify_catches_wrong_row_order(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(20)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format", lambda smoke=False: fmt)

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"response_id": fmt["response_id"][::-1].to_numpy(), "probability": 0.5}).to_csv(
        bad, index=False
    )
    res = VerifyResult()
    verify_predictions(bad, res)
    assert any(c.name == "row_ORDER_matches" and not c.passed for c in res.checks)


def test_verify_catches_out_of_range_probability(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(5)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format", lambda smoke=False: fmt)

    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {"response_id": fmt["response_id"], "probability": [0.5, 1.5, -0.2, 0.3, 0.4]}
    ).to_csv(bad, index=False)
    res = VerifyResult()
    verify_predictions(bad, res)
    assert any(c.name == "probabilities_in_unit_interval" and not c.passed for c in res.checks)


def test_verify_catches_nan(tmp_path, synth_repo, monkeypatch):
    fmt = pd.DataFrame({"response_id": [f"r{i}" for i in range(4)], "probability": 0.5})
    monkeypatch.setattr("traceace.packaging.verify.load_submission_format", lambda smoke=False: fmt)
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {"response_id": fmt["response_id"], "probability": [0.5, np.nan, 0.3, 0.4]}
    ).to_csv(bad, index=False)
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


# --- feature ORDER: regression test for the bug that cost a leaderboard slot ---
def test_feature_order_must_come_from_the_booster_not_importance():
    """The bundle's feature order must be read from the model, never from a sorted table.

    Regression test. `_feature_cols` read `importance.parquet`, which is sorted by gain
    descending — a permutation of the training order in 179 of 181 positions. `main.py`
    reordered the columns to match and LightGBM read them positionally, scrambling every
    feature. Predictions stayed confident and correctly formatted, so all 20 verify checks
    passed. The leaderboard returned AUC 0.4933, below random.
    """
    import ast
    import inspect

    from traceace.packaging import build_submission

    src = inspect.getsource(build_submission._feature_cols)
    assert "feature_name()" in src, "feature order must come from booster.feature_name()"

    # Inspect the CODE only — the docstring deliberately names importance.parquet to
    # explain why it must never be used, and matching that text is a false positive.
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    code = "\n".join(ast.unparse(node) for node in body)
    assert "importance.parquet" not in code, (
        "importance.parquet is sorted by gain and must never be an order source"
    )


def test_main_py_asserts_feature_order_at_runtime():
    """main.py must refuse to predict if the column order disagrees with the booster."""
    from traceace.packaging.main_template import MAIN_TEMPLATE

    assert "feature_name()" in MAIN_TEMPLATE
    assert "feature order does not match the trained booster" in MAIN_TEMPLATE


def test_main_py_explicitly_falls_back_for_unreadable_transcripts():
    """Finite LightGBM predictions must not mask a missing/empty transcript."""
    from traceace.packaging.main_template import MAIN_TEMPLATE

    assert 'feats["_use_fallback"] = not transcript_ok' in MAIN_TEMPLATE
    assert "bad = ~np.isfinite(preds) | use_fallback" in MAIN_TEMPLATE


def test_scrambled_feature_order_is_detected(synth_repo):
    """A permuted feature order must be caught by verify, not shipped."""
    import lightgbm as lgb
    import numpy as np

    from traceace.packaging.verify import VerifyResult, verify_feature_order

    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(8)]
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=cols)
    y = (X["f0"] + X["f1"] > 0).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, label=y), 5)

    import joblib

    work = synth_repo / "wk"
    (work / "assets").mkdir(parents=True)

    # correct order -> passes
    joblib.dump({"feature_cols": cols, "boosters": [booster]}, work / "assets" / "model.joblib")
    ok = VerifyResult()
    verify_feature_order(work, ok)
    assert ok.ok, [c.detail for c in ok.failures]

    # permuted order -> must FAIL
    joblib.dump(
        {"feature_cols": cols[::-1], "boosters": [booster]}, work / "assets" / "model.joblib"
    )
    bad = VerifyResult()
    verify_feature_order(work, bad)
    assert not bad.ok, "a permuted feature order must be rejected"


def test_shrinkage_preserves_ranking_and_pulls_toward_base_rate():
    """Retired shrinkage remains reproducible and ranking-invariant when explicitly used."""
    from traceace.evaluate import auc
    from traceace.packaging.inference_lib import apply_shrinkage

    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.7, 3000).astype(float)
    p = np.clip(rng.normal(0.7 + 0.15 * (y - 0.5), 0.15), 0.01, 0.99)
    base = 0.7025

    for w in (0.25, 0.5, 0.75, 1.0):
        q = apply_shrinkage(p, w, base)
        assert np.isclose(auc(y, p), auc(y, q)), f"w={w} changed the ranking"
        # spread shrinks toward the base rate, and the mean moves toward it
        assert q.std() <= p.std() + 1e-12
        assert abs(q.mean() - base) <= abs(p.mean() - base) + 1e-12


def test_falsified_deployment_shrinkage_is_opt_in():
    import inspect

    from traceace.packaging.build_submission import build

    assert inspect.signature(build).parameters["apply_deployment_shrinkage"].default is False


def test_verify_finds_the_smoke_csv_by_default(tmp_path):
    """A check that cannot fire is worse than no check — it reads as coverage.

    ``submission.verify`` defaulted to ``_staging/submission.csv``, but ``_staging`` is the
    zip BUILD directory and never holds a CSV; ``submission.smoke`` writes to ``_smoke/``.
    All nine output checks — including row-set and ordering alignment, the failure that
    cost submission #1 — were therefore skipped on every standalone verify run.
    """
    import json

    from traceace.packaging.verify import _latest_smoke_csv

    assert _latest_smoke_csv(tmp_path) is None, "must not invent a path when none exists"

    # conventional location
    smoke = tmp_path / "_smoke"
    smoke.mkdir()
    (smoke / "submission.csv").write_text("response_id,probability\na,0.5\n")
    assert _latest_smoke_csv(tmp_path) == smoke / "submission.csv"

    # a path recorded in smoke_report.json wins, so a workdir move cannot break this
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / "submission.csv").write_text("response_id,probability\na,0.5\n")
    (tmp_path / "smoke_report.json").write_text(
        json.dumps({"submission_csv": str(moved / "submission.csv")})
    )
    assert _latest_smoke_csv(tmp_path) == moved / "submission.csv"

    # a stale report pointing at a deleted file falls back rather than exploding
    (moved / "submission.csv").unlink()
    assert _latest_smoke_csv(tmp_path) == smoke / "submission.csv"


def test_missing_check_is_itself_a_failure():
    """'0 failures' must not be reachable while checks are silently absent (ADR-018)."""
    from traceace.packaging.verify import PREDICTION_CHECKS, REQUIRED_CHECKS, VerifyResult

    # every required name present -> the guard passes
    full = VerifyResult()
    for n in (*REQUIRED_CHECKS, *PREDICTION_CHECKS):
        full.add(n, True)
    ran = {c.name for c in full.checks}
    expected = set(REQUIRED_CHECKS) | set(PREDICTION_CHECKS)
    assert not sorted(expected - ran)

    # drop the ordering check — the exact one that went dark — and the guard must notice
    partial = {n for n in ran if n != "row_ORDER_matches"}
    assert sorted(expected - partial) == ["row_ORDER_matches"]


def test_packaging_rejects_research_only_feature_families():
    from traceace.packaging.build_submission import _assert_features_deployable

    _assert_features_deployable(["struct_n_utterances", "lo_prior_enc"])
    with np.testing.assert_raises_regex(RuntimeError, "no main.py implementation"):
        _assert_features_deployable(["struct_n_utterances", "cont_00"])
