"""Tutoring-move classifier — LLM-quality taxonomy at encoder-class inference cost.

Trained on the annotations from :mod:`traceace.annotate` (dev-time LLM or heuristic
labels), this small classifier labels **every** utterance in **every** transcript cheaply
enough to run at inference. That is the whole point of the annotate → distil design: we
get an LLM-derived taxonomy without putting a generative model in the submission path.

Two backends, same interface:

* ``tfidf`` (default, CPU, zero units): TF-IDF + linear model. Surprisingly strong for
  short dialogue acts, and it keeps the self-test GPU-free.
* ``encoder`` (L4): a small transformer fine-tuned for sequence classification, when the
  ladder justifies the spend.

Downstream, ``features.moves`` turns per-utterance labels into session- and
LO-conditioned move-distribution features, which is the substrate for the
"which tutoring moves are most effective" research question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..annotate import STUDENT_MOVES, TUTOR_MOVES, annotations_path
from ..config import get_config
from ..logging_utils import get_logger
from ..paths import models_dir, runs_dir
from ..progress import heartbeat
from ..tasks import task

log = get_logger("model.move_classifier")

VERSION = "v1"
ALL_MOVES = TUTOR_MOVES + STUDENT_MOVES


def model_path(
    backend: str = "tfidf",
    annotation_backend: str = "heuristic",
    subsample: int | None = None,
) -> Path:
    suffix = "" if subsample is None else f"_sub{subsample}"
    return models_dir() / (
        f"move_classifier_{backend}_{annotation_backend}_{VERSION}{suffix}.joblib"
    )


def _grouped_split(ann: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by session so transcript-specific wording cannot leak into evaluation."""
    from sklearn.model_selection import GroupShuffleSplit

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    if ann["session_id"].nunique() < 2:
        raise ValueError("move-classifier evaluation needs at least two sessions")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(ann, groups=ann["session_id"]))
    return np.asarray(train_idx), np.asarray(test_idx)


def _build_tfidf_pipeline(seed: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=seed)),
        ]
    )


@task(
    "model.move_classifier",
    requires="cpu",
    max_tier="cpu",
    description="train a cheap classifier on move annotations (runs over all transcripts)",
)
def train(
    force: bool = False,
    subsample: int | None = None,
    backend: str = "tfidf",
    annotation_backend: str = "heuristic",
    test_size: float = 0.2,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train and evaluate the move classifier on held-out annotations.

    The role is prepended to the text so one model handles both tutor and student move
    vocabularies without conflating them.
    """
    import joblib
    from sklearn.metrics import classification_report

    if backend != "tfidf":
        raise NotImplementedError(
            f"backend={backend!r} not implemented yet; the encoder backend is an L4 task "
            "reserved for after the cheap ladder. Use backend='tfidf'."
        )

    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    apath = annotations_path(annotation_backend, subsample)
    if not apath.is_file():
        raise FileNotFoundError(f"{apath} missing — run tasks.run('annotate.moves') first")
    ann = pd.read_parquet(apath)
    if subsample is not None:
        ann = ann.head(max(50, subsample))

    X = (ann["role"].astype(str) + " || " + ann["content"].astype(str)).to_numpy()
    y = ann["move"].astype(str).to_numpy()
    train_idx, test_idx = _grouped_split(ann, test_size, seed)
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    pipe = _build_tfidf_pipeline(seed)
    with heartbeat("fit move classifier"):
        pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    acc = float((pred == yte).mean())
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)

    mp = model_path(backend, annotation_backend, subsample)
    mp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "classes": sorted(set(y))}, mp)

    res = {
        "output_path": str(mp),
        "backend": backend,
        "annotation_backend": annotation_backend,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "accuracy": acc,
        "macro_f1": float(rep.get("macro avg", {}).get("f1-score", float("nan"))),
        "n_classes": int(len(set(y))),
    }
    d = runs_dir() / "move_classifier"
    d.mkdir(parents=True, exist_ok=True)
    report_name = f"{backend}_{annotation_backend}{'' if subsample is None else f'_sub{subsample}'}"
    (d / f"{report_name}.json").write_text(
        json.dumps({**res, "report": rep}, indent=2, default=str)
    )
    log.info("move_classifier[%s]: acc=%.4f macro_f1=%.4f", backend, acc, res["macro_f1"])
    return res


def load_classifier(backend: str = "tfidf", annotation_backend: str = "heuristic"):
    import joblib

    mp = model_path(backend, annotation_backend)
    if not mp.is_file():
        raise FileNotFoundError(f"{mp} missing — run tasks.run('model.move_classifier')")
    return joblib.load(mp)


def predict_moves(
    roles: list[str],
    texts: list[str],
    backend: str = "tfidf",
    annotation_backend: str = "heuristic",
) -> np.ndarray:
    """Label utterances with move types. Used by features.moves."""
    bundle = load_classifier(backend, annotation_backend)
    X = np.array([f"{r} || {t}" for r, t in zip(roles, texts)])
    return bundle["pipeline"].predict(X)
