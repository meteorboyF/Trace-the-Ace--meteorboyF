"""Train/serve parity — the shipped feature code must equal the training feature code.

``inference_lib.py`` is copied verbatim into submission.zip and is what actually runs on
the competition's test set. If it ever drifts from the training-time blocks in
``features/``, the model receives differently-computed inputs than it was trained on and
the leaderboard score silently collapses — with no error anywhere.

These tests compute both implementations on the same synthetic transcript and assert
they agree value-for-value.
"""

from __future__ import annotations

import numpy as np

from traceace.features.common import normalize_frame as train_normalize
from traceace.features.linguistic import session_linguistic_features
from traceace.features.structural import session_structural_features
from traceace.features.temporal import session_temporal_features
from traceace.packaging import inference_lib as ilib


def _both(synth_transcript):
    train_df = train_normalize(synth_transcript.copy())
    infer_df = ilib.normalize_frame(synth_transcript.copy())
    return train_df, infer_df


def test_structural_parity(synth_repo, synth_transcript):
    train_df, infer_df = _both(synth_transcript)
    a = session_structural_features("sess0000", train_df)
    a.pop("session_id")
    b = ilib.structural_features(infer_df)
    assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
    for k in a:
        assert np.isclose(a[k], b[k], equal_nan=True), f"{k}: train={a[k]} infer={b[k]}"


def test_linguistic_parity(synth_repo, synth_transcript):
    train_df, infer_df = _both(synth_transcript)
    a = session_linguistic_features("sess0000", train_df)
    a.pop("session_id")
    b = ilib.linguistic_features(infer_df)
    assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
    for k in a:
        assert np.isclose(a[k], b[k], equal_nan=True), f"{k}: train={a[k]} infer={b[k]}"


def test_temporal_parity(synth_repo, synth_transcript):
    train_df, infer_df = _both(synth_transcript)
    a = session_temporal_features("sess0000", train_df)
    a.pop("session_id")
    b = ilib.temporal_features(infer_df)
    assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
    for k in a:
        assert np.isclose(a[k], b[k], equal_nan=True), f"{k}: train={a[k]} infer={b[k]}"


def test_window_construction_parity(synth_repo, synth_transcript):
    from traceace.features.lo_alignment import _windows

    train_df, infer_df = _both(synth_transcript)
    assert _windows(train_df) == ilib.windows(infer_df)


def test_timestamp_parsing_parity():
    from traceace.data import parse_elapsed_seconds

    for ts in ["0:00:00", "1:23:45", "0:05:09", "12:00:01", "bad", None]:
        a = parse_elapsed_seconds(ts)
        b = ilib.parse_elapsed_seconds(ts)
        assert (np.isnan(a) and np.isnan(b)) or a == b


def test_inference_lib_has_no_traceace_imports():
    """The shipped library must be standalone — it runs from the zip with no package."""
    from pathlib import Path

    src = Path(ilib.__file__).read_text()
    assert "from traceace" not in src and "import traceace" not in src


def test_inference_lib_never_prints():
    from pathlib import Path

    src = Path(ilib.__file__).read_text()
    assert "print(" not in src, "inference_lib must not print — it handles test data"
