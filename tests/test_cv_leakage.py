"""The most important test in the repo: folds must never leak a session.

One session yields up to 10 response rows sharing an identical transcript. If folds
split by response, the same transcript appears in train and validation and every score
becomes fiction. This test asserts zero session overlap and runs in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from traceace.cv import assign_folds


def test_no_session_appears_in_two_folds(synth_repo, synth_train):
    folds = assign_folds(synth_train, n_splits=5, seed=1234)
    per_session = folds.groupby("session_id")["fold"].nunique()
    assert (per_session == 1).all(), (
        "LEAKAGE: a session was split across folds — CV would be invalid"
    )


def test_train_and_val_sessions_are_disjoint(synth_repo, synth_train):
    folds = assign_folds(synth_train, n_splits=5, seed=1234)
    for k in sorted(folds["fold"].unique()):
        tr = set(folds.loc[folds["fold"] != k, "session_id"])
        va = set(folds.loc[folds["fold"] == k, "session_id"])
        assert tr.isdisjoint(va), f"fold {k}: {len(tr & va)} sessions in both splits"


def test_every_row_assigned(synth_repo, synth_train):
    folds = assign_folds(synth_train, n_splits=5, seed=1234)
    assert (folds["fold"] >= 0).all()
    assert len(folds) == len(synth_train)


def test_multi_response_sessions_stay_together(synth_repo, synth_train):
    """Responses from one session must land in the same fold — the core invariant."""
    folds = assign_folds(synth_train, n_splits=5, seed=1234)
    multi = folds.groupby("session_id").filter(lambda g: len(g) > 1)
    assert len(multi) > 0, "fixture should contain multi-response sessions"
    assert multi.groupby("session_id")["fold"].nunique().eq(1).all()


def test_folds_are_deterministic(synth_repo, synth_train):
    a = assign_folds(synth_train, n_splits=5, seed=1234)
    b = assign_folds(synth_train, n_splits=5, seed=1234)
    pd.testing.assert_series_equal(a["fold"], b["fold"])


def test_missing_group_column_raises(synth_repo, synth_train):
    with pytest.raises(KeyError):
        assign_folds(synth_train.drop(columns=["session_id"]), n_splits=5, seed=1)
