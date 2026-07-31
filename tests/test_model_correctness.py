"""Regression tests for silent model/evaluation correctness failures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traceace.calibration import _persist_calibrator
from traceace.evaluate import align_compatible_oof
from traceace.features import assemble
from traceace.io import LABEL_COL
from traceace.models.gbdt import LO_COL, _fold_safe_lo_encoding, _smoothed_map
from traceace.packaging.inference_lib import lo_prior_values
from traceace.packaging.verify import (
    MIN_TRAIN_SANITY_AUC,
    VerifyResult,
    verify_no_cross_row_features,
    verify_prediction_sanity,
)


def test_training_auc_gate_detects_corruption_without_requiring_objective_prior():
    assert 0.4738 < MIN_TRAIN_SANITY_AUC < 0.7466


def _target_encoding_frame() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for j in range(30):
            # Make fold 0 deliberately different so an outer-fold overwrite is obvious.
            label = 1.0 if fold == 0 else float(j % 2)
            rows.append(
                {
                    "response_id": f"r{fold}_{j}",
                    "session_id": f"s{fold}_{j}",
                    LO_COL: "shared_lo",
                    "fold": fold,
                    LABEL_COL: label,
                }
            )
    return pd.DataFrame(rows)


def test_outer_fold_target_encoding_never_uses_validation_labels():
    """Changing outer-validation labels must not change any feature seen by that model.

    Regression test: one shared encoding array was filled once per outer fold. Later folds
    overwrote earlier folds' values, so folds 0--3 received encodings built while their
    validation labels were present. The OOF score looked valid but was leaked.
    """
    frame = _target_encoding_frame()
    encoded = _fold_safe_lo_encoding(frame, smoothing=20.0, seed=1234, outer_fold=0)

    changed = frame.copy()
    changed.loc[changed["fold"] == 0, LABEL_COL] = 0.0
    encoded_after_validation_mutation = _fold_safe_lo_encoding(
        changed, smoothing=20.0, seed=1234, outer_fold=0
    )

    assert np.allclose(encoded, encoded_after_validation_mutation)

    train = frame[frame["fold"] != 0]
    expected_map, _ = _smoothed_map(train, smoothing=20.0)
    validation = frame["fold"].eq(0).to_numpy()
    assert np.allclose(encoded[validation], float(expected_map["shared_lo"]))


def test_selecting_no_calibration_removes_stale_calibrator(tmp_path):
    """A later 'none' result must not leave an older Platt/isotonic file shippable."""
    import joblib

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump({"method": "platt", "coef": 3.0}, model_dir / "calibrator_plain.joblib")
    joblib.dump({"method": "platt", "model": object()}, model_dir / "calibrator.joblib")

    _persist_calibrator("none", None, model_dir)

    assert not (model_dir / "calibrator_plain.joblib").exists()
    assert not (model_dir / "calibrator.joblib").exists()


def test_prediction_sanity_runs_the_requested_archive(tmp_path, monkeypatch):
    """An alternate candidate must not borrow a PASS from submission.zip."""
    from zipfile import ZIP_DEFLATED, ZipFile

    features = pd.DataFrame(
        {
            "response_id": [f"r{i}" for i in range(100)],
            "session_id": [f"s{i}" for i in range(100)],
        }
    )
    labels = pd.DataFrame(
        {
            "response_id": features["response_id"],
            LABEL_COL: [float(i % 2) for i in range(100)],
        }
    )
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()

    monkeypatch.setattr("traceace.packaging.verify.submission_dir", lambda: tmp_path)
    monkeypatch.setattr("traceace.io.load_train_features", lambda: features)
    monkeypatch.setattr("traceace.io.load_train_labels", lambda: labels)
    monkeypatch.setattr("traceace.paths.transcripts_dir", lambda: transcript_dir)

    good_main = """
from pathlib import Path
import pandas as pd
here = Path(__file__).parent
df = pd.read_csv(here / "data" / "test_features.csv")
df["probability"] = [0.99 if int(v[1:]) % 2 else 0.01 for v in df["response_id"]]
df[["response_id", "probability"]].to_csv(here / "submission.csv", index=False)
"""
    bad_main = "raise SystemExit(7)\n"
    default_zip = tmp_path / "submission.zip"
    candidate_zip = tmp_path / "candidate.zip"
    with ZipFile(default_zip, "w", ZIP_DEFLATED) as zf:
        zf.writestr("main.py", good_main)
    with ZipFile(candidate_zip, "w", ZIP_DEFLATED) as zf:
        zf.writestr("main.py", bad_main)

    result = VerifyResult()
    verify_prediction_sanity(result, zip_path=candidate_zip)

    assert any(
        check.name == "prediction_sanity" and not check.passed and "exited 7" in check.detail
        for check in result.checks
    )


def test_feature_assembly_rejects_a_different_subsample_cohort(monkeypatch):
    base = pd.DataFrame(
        {
            "response_id": ["r1", "r2"],
            "session_id": ["selected_1", "selected_2"],
            LABEL_COL: [0.0, 1.0],
        }
    )
    wrong_block = pd.DataFrame(
        {
            "session_id": ["different_1", "different_2"],
            "struct_n_utterances": [10.0, 20.0],
        }
    )
    monkeypatch.setattr(assemble, "load_block", lambda name, subsample=None: wrong_block)

    with np.testing.assert_raises_regex(RuntimeError, "different session cohorts"):
        assemble.build_matrix(base, blocks=["structural"], subsample=2)


def test_each_booster_uses_its_own_lo_prior_map():
    lo_ids = np.asarray(["seen", "unseen"])
    fold0 = {"fallback": 0.2, "values": {"seen": 0.3}}
    fold1 = {"fallback": 0.7, "values": {"seen": 0.8}}

    assert np.allclose(lo_prior_values(lo_ids, fold0), [0.3, 0.2])
    assert np.allclose(lo_prior_values(lo_ids, fold1), [0.8, 0.7])


def test_submission_verifier_always_rejects_cross_row_features(tmp_path):
    import joblib

    work = tmp_path / "work"
    (work / "assets").mkdir(parents=True)
    joblib.dump(
        {"feature_cols": ["lopos_n_competing_los"]},
        work / "assets" / "model.joblib",
    )
    result = VerifyResult()
    verify_no_cross_row_features(work, result)
    assert not result.ok


def test_equal_length_but_different_oof_cohorts_are_rejected():
    reference = pd.DataFrame(
        {
            "response_id": ["a", "b"],
            "session_id": ["sa", "sb"],
            LABEL_COL: [0.0, 1.0],
            "pred": [0.1, 0.9],
        }
    )
    candidate = pd.DataFrame(
        {
            "response_id": ["c", "d"],
            "session_id": ["sc", "sd"],
            LABEL_COL: [0.0, 1.0],
            "pred": [0.2, 0.8],
        }
    )
    with np.testing.assert_raises_regex(RuntimeError, "cohorts differ"):
        align_compatible_oof(reference, candidate, "reference", "candidate")


def test_ensemble_requires_an_explicit_compatible_experiment_list(synth_repo):
    from traceace.ensemble import blend

    with np.testing.assert_raises_regex(ValueError, "must be explicit"):
        blend(experiments=None)


def test_embedding_consumers_resolve_the_producer_cache(synth_repo):
    """Content/semantic alignment must read the exact path the GPU producer writes."""
    from traceace.features.content import _embedding_paths
    from traceace.features.window_embeddings import (
        lo_embedding_path,
        window_embedding_path,
    )

    model = "vendor/model"
    assert _embedding_paths(model, 17) == (
        window_embedding_path(model, 17),
        lo_embedding_path(model, 17),
    )
    assert lo_embedding_path(model, 17) != lo_embedding_path(model, None)


def test_content_pca_is_namespaced_by_cohort_and_pooling(synth_repo):
    """A smoke PCA or alternate top-k must never overwrite the full-data transform."""
    from traceace.features.content import pca_path

    full = pca_path("model", 48, 3, None, "source")
    smoke = pca_path("model", 48, 3, 400, "source")
    other_pool = pca_path("model", 48, 5, None, "source")
    assert len({full, smoke, other_pool}) == 3


def test_fold_safe_content_pca_never_fits_validation_covariates():
    """Changing validation rows cannot alter the representation of training rows."""
    from traceace.models.gbdt import _fold_safe_pca

    rng = np.random.default_rng(123)
    raw = rng.normal(size=(20, 6)).astype(np.float32)
    train = np.zeros(20, dtype=bool)
    train[:15] = True
    validation = ~train

    original = _fold_safe_pca(raw, train, validation, n_components=3, seed=7)
    perturbed = raw.copy()
    perturbed[validation] += 10_000
    changed = _fold_safe_pca(perturbed, train, validation, n_components=3, seed=7)

    np.testing.assert_allclose(original[train], changed[train], atol=1e-6)
    assert not np.allclose(original[validation], changed[validation])


def test_semantic_alignment_has_an_explicit_nonproduction_cache_alias():
    """Research BGE alignment must never silently replace production lexical features."""
    from traceace.features.assemble import BLOCKS

    assert BLOCKS["lo_alignment"][1] == "v1_lexical"
    assert BLOCKS["lo_alignment_embedding"][1] == "v1_embedding"
    assert BLOCKS["lo_alignment"][0] == BLOCKS["lo_alignment_embedding"][0]


def test_bge_attention_masks_padding_and_emits_one_logit_per_response():
    torch = pytest.importorskip("torch")

    from traceace.models.bge_attention import WindowAttention

    torch.manual_seed(7)
    model = WindowAttention.build(dim=8, hidden=6, dropout=0.0, base_rate=0.7)
    model.eval()
    windows = torch.randn(2, 4, 8)
    query = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    positions = torch.rand(2, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    with torch.no_grad():
        original = model(windows, query, positions, mask)
        changed = windows.clone()
        changed[0, 2:] = 1_000_000
        replay = model(changed, query, positions, mask)
    assert original.shape == (2,)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original[0], replay[0])


def test_move_classifier_holdout_is_session_disjoint():
    from traceace.models.move_classifier import _grouped_split

    ann = pd.DataFrame(
        {
            "session_id": np.repeat([f"s{i}" for i in range(10)], 3),
            "move": ["m"] * 30,
        }
    )
    train_idx, test_idx = _grouped_split(ann, test_size=0.2, seed=1234)
    assert set(ann.iloc[train_idx]["session_id"]).isdisjoint(set(ann.iloc[test_idx]["session_id"]))


def test_move_classifier_artifacts_include_annotation_source(synth_repo):
    from traceace.models.move_classifier import model_path

    assert model_path("tfidf", "heuristic") != model_path("tfidf", "vllm")
    assert model_path("tfidf", "heuristic") != model_path("tfidf", "heuristic", 50)


def test_move_annotations_are_subsample_namespaced(synth_repo):
    from traceace.annotate import annotations_path

    assert annotations_path("heuristic") != annotations_path("heuristic", 50)


def test_shrinkage_gain_is_crossfit_and_excludes_overlapping_rows():
    from traceace.unseen_lo import _crossfit_shrinkage

    draws = [
        (
            np.asarray(["shared", "a"]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.9, 0.4]),
            0.5,
        ),
        (
            np.asarray(["shared", "b"]),
            np.asarray([1.0, 1.0]),
            np.asarray([0.9, 0.6]),
            0.5,
        ),
        (
            np.asarray(["c", "d"]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.2, 0.8]),
            0.5,
        ),
    ]
    y, pred = _crossfit_shrinkage(draws)
    assert len(y) == len(pred) == 6
    assert np.isfinite(pred).all()
