"""Fast, deployable supervised text model over targeted tutoring dialogue.

Unlike the failed end-to-end transformer, this model observes all sessions, focuses on
objective-relevant windows without consuming objective text as a predictor, and uses
fixed hashing so train/serve vocabulary drift is impossible. Session-grouped OOF
predictions make it eligible only as a guarded complement to the established GBDT.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..cv import load_folds
from ..evaluate import experiment_dir, load_oof, save_oof, score_frame
from ..features.common import normalize_frame
from ..features.lo_alignment import fit_lo_vectorizer
from ..io import LABEL_COL, load_train_features
from ..logging_utils import get_logger
from ..packaging.inference_lib import (
    topk_spans,
    window_texts,
    windows,
)
from ..packaging.sparse_text_lib import sparse_text_document
from ..paths import transcripts_dir
from ..progress import pbar
from ..staging import stage_local
from ..tasks import task

log = get_logger("model.sparse_text")


def _features():
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            (
                "word",
                HashingVectorizer(
                    n_features=2**16,
                    ngram_range=(1, 2),
                    lowercase=True,
                    alternate_sign=False,
                    norm="l2",
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                HashingVectorizer(
                    n_features=2**16,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    alternate_sign=False,
                    norm="l2",
                ),
            ),
        ]
    )


def _classifier(seed: int, alpha: float):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=0.05,
        max_iter=60,
        tol=1e-3,
        average=True,
        random_state=seed,
    )


def _pipeline(seed: int, alpha: float):
    from sklearn.pipeline import Pipeline

    return Pipeline([("features", _features()), ("classifier", _classifier(seed, alpha))])


def _documents(frame: pd.DataFrame, max_chars: int, context_utterances: int) -> list[str]:
    """Render one independent document per response, sharing only transcript reads."""
    selector = fit_lo_vectorizer()
    documents: dict[str, str] = {}
    tdir = transcripts_dir()
    groups = frame.groupby("session_id", sort=False)
    for sid, responses in pbar(groups, desc="sparse text documents", unit="session"):
        path = tdir / f"{sid}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing transcript for training session {sid}")
        transcript = normalize_frame(pd.read_csv(path, dtype=str))
        spans = windows(transcript)
        matrix = selector.transform(window_texts(transcript, spans))
        for row in responses.itertuples(index=False):
            selected = topk_spans(str(row.learning_objective), selector, matrix, spans)
            # Materialize the shared function here so parity tests can compare it directly.
            documents[str(row.response_id)] = sparse_text_document(
                transcript,
                selected,
                max_chars=max_chars,
                context_utterances=context_utterances,
            )
    return [documents[str(response_id)] for response_id in frame["response_id"]]


@task(
    "model.sparse_text",
    requires="cpu",
    max_tier="cpu",
    description="session-safe hashed word+character text model over targeted dialogue",
)
def train(
    force: bool = False,
    subsample: int | None = None,
    experiment: str = "model.sparse_text",
    alpha: float = 1e-3,
    max_chars: int = 8000,
    context_utterances: int = 96,
    cv_seed: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    import joblib

    from ..config import get_config

    stage_local()
    cfg = get_config()
    seed = int(cfg.seed if seed is None else seed)
    model_dir = experiment_dir(experiment, subsample, cv_seed)
    run_config = {
        "experiment": experiment,
        "subsample": subsample,
        "cv_seed": cv_seed,
        "alpha": alpha,
        "max_chars": max_chars,
        "context_utterances": context_utterances,
        "seed": seed,
    }
    manifest_path = model_dir / "training_manifest.json"
    model_path = model_dir / "deployment.joblib"
    if manifest_path.is_file() and model_path.is_file() and not force:
        cached = json.loads(manifest_path.read_text())
        if cached.get("run_config") != run_config:
            raise RuntimeError(
                "sparse-text artifacts use different hyperparameters; choose a new "
                "experiment name or pass force=True explicitly"
            )
        result = score_frame(
            load_oof(experiment, subsample=subsample, cv_seed=cv_seed),
            experiment,
            subsample=subsample,
            cv_seed=cv_seed,
        )
        return {**result, "cached": True, "output_path": str(model_dir)}
    folds = load_folds(subsample=subsample, cv_seed=cv_seed)
    metadata = load_train_features()[["response_id", "session_id", "learning_objective"]].copy()
    frame = folds.merge(
        metadata, on=["response_id", "session_id"], how="left", validate="one_to_one"
    )
    if frame["learning_objective"].isna().any():
        raise RuntimeError("sparse text training has missing learning-objective metadata")
    docs = np.asarray(_documents(frame, max_chars, context_utterances), dtype=object)
    # Hashing is stateless: transform once and reuse the sparse matrix across all five
    # fold fits. Re-vectorizing long transcripts per fold costs minutes and several large
    # transient allocations without changing a single feature value.
    features = _features()
    matrix = features.fit_transform(docs.tolist())
    y = frame[LABEL_COL].to_numpy(dtype=int)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in frame["fold"].unique()):
        train_mask = frame["fold"].ne(fold).to_numpy()
        valid_mask = ~train_mask
        model = _classifier(seed + fold, alpha)
        model.fit(matrix[train_mask], y[train_mask])
        oof[valid_mask] = model.predict_proba(matrix[valid_mask])[:, 1]
        fold_metrics.append(
            {"fold": fold, "n_train": int(train_mask.sum()), "n_valid": int(valid_mask.sum())}
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("sparse text model left OOF rows unpredicted")

    scored = frame[["response_id", "session_id", LABEL_COL]].copy()
    scored["pred"] = oof
    save_oof(experiment, scored, subsample=subsample, cv_seed=cv_seed)

    # Deployment model is fitted only after honest OOF evaluation; no fold model baggage.
    deployment_classifier = _classifier(seed, alpha)
    deployment_classifier.fit(matrix, y)
    from sklearn.pipeline import Pipeline

    deployment = Pipeline([("features", features), ("classifier", deployment_classifier)])
    model_dir.mkdir(parents=True, exist_ok=True)
    temp = model_dir / ".deployment.joblib.tmp"
    joblib.dump(deployment, temp, compress=3)
    temp.replace(model_path)
    manifest = {
        "run_config": run_config,
        "experiment": experiment,
        "subsample": subsample,
        "cv_seed": cv_seed,
        "alpha": alpha,
        "max_chars": max_chars,
        "context_utterances": context_utterances,
        "objective_text_is_predictor": False,
        "fold_metrics": fold_metrics,
    }
    (model_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    result = score_frame(scored, experiment, subsample=subsample, cv_seed=cv_seed)
    result.update({"output_path": str(model_dir), "model_path": str(model_path), **manifest})
    log.info("sparse text: logloss=%.5f auc=%.4f", result["logloss"], result["auc"])
    return result
