"""Learning-objective difficulty that survives an *unseen* objective.

**The gap this closes.** ``model.gbdt`` encodes objective difficulty as a per-objective
empirical rate (``lo_prior_enc``). That lookup is worth AUC 0.706 when the objective was in
training and **exactly 0.500 when it was not**, because an unseen objective falls back to the
global rate. The test set contains objectives absent from training (organiser forum ruling,
2026-07-27), and our leaderboard AUROC of 0.6014 matches our *transcript-only* CV AUROC
almost exactly — the lookup is contributing close to nothing there (docs/ENDGAME.md §2).

**The fix.** Objectives are described in text, and that text generalises: a model reading
"divide a fraction by a whole number" can place its difficulty without ever having seen that
objective's outcomes. Measured under objective-grouped folds with AUC computed *within* fold:

===========================================  ==========
signal                                       AUC
===========================================  ==========
per-objective lookup, objective unseen       0.500
**objective text -> difficulty (this)**      **0.575**
transcript-only model                        0.606
transcript + objective text, blended         0.612
===========================================  ==========

So this is not a replacement for the transcript — it is the component that keeps working when
the lookup dies, worth roughly +0.007 AUROC on top of dialogue signal.

**On the anti-goal.** The organisers' stated anti-goal is predicting correctness from inferred
objective difficulty *without reference to the transcript*. A model that uses both is not the
anti-goal, and every report still headlines the delta over ``baseline.lo_only``. What would be
dishonest is letting this feature stand in for dialogue signal, which is precisely what
``interpret.ablation_repeated`` under objective folds is for.

**Leakage.** This feature is built from labels, so it is fold-safe by construction, exactly
like ``gbdt._fold_safe_lo_encoding``:

* **Validation rows** are scored by a model fitted only on the outer training fold.
* **Training rows** are scored by an *inner* objective-grouped K-fold, so the booster does not
  learn to trust an estimate that was fitted on the very row it is predicting. Skipping this
  inner loop makes the feature look far more reliable at training time than it is at
  validation time — the mistake ADR-014 records for the lookup.

The inner split groups by **objective**, not session: the deployment question is "what does
this text predict for an objective I have never scored?", so the inner estimate has to be
out-of-objective too, or it measures memorisation again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..io import LABEL_COL
from ..logging_utils import get_logger

log = get_logger("features.lo_difficulty")

LO_COL = "learning_objective_id"
LO_TEXT_COL = "learning_objective"
LO_TEXT_ENC_COL = "lo_text_difficulty"

# Word 1-2 grams beat char n-grams on held-out objectives (0.575 vs 0.560) and stay readable,
# which matters: an education researcher can inspect the coefficients and see which topic
# vocabulary predicts difficulty. That is a reportable finding, not just a feature.
DEFAULT_NGRAM = (1, 2)
DEFAULT_MIN_DF = 2
DEFAULT_C = 1.0
INNER_SPLITS = 5


@dataclass
class LoTextDifficultyModel:
    """A fitted text -> difficulty model, serialisable for the submission path."""

    vectorizer: Any
    classifier: Any
    fallback: float

    def predict(self, lo_texts: pd.Series) -> np.ndarray:
        """Difficulty estimate for each objective description."""
        if len(lo_texts) == 0:
            return np.zeros(0, dtype=float)
        texts = lo_texts.fillna("").astype(str)
        blank = texts.str.strip().eq("").to_numpy()
        out = self.classifier.predict_proba(self.vectorizer.transform(texts))[:, 1]
        # An objective with no usable description gets the training base rate rather than
        # whatever the empty-vector intercept happens to be.
        out[blank] = self.fallback
        return out.astype(float)


def fit_lo_text_difficulty(
    frame: pd.DataFrame,
    seed: int,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM,
    min_df: int = DEFAULT_MIN_DF,
    c: float = DEFAULT_C,
) -> LoTextDifficultyModel:
    """Fit objective-text -> difficulty on ``frame``.

    Fitted on **one row per objective** rather than per response. Per-response fitting weights
    each objective by how often it was assessed, which teaches the model the training
    frequency distribution — a property of our sample, not of difficulty, and one that will
    not hold in a different deployment.
    """
    required = {LO_COL, LO_TEXT_COL, LABEL_COL}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"lo_text_difficulty needs columns {sorted(missing)}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    per_objective = (
        frame.groupby(LO_COL)
        .agg(text=(LO_TEXT_COL, "first"), rate=(LABEL_COL, "mean"), n=(LABEL_COL, "size"))
        .reset_index()
    )
    fallback = float(frame[LABEL_COL].mean())

    # Regression on the per-objective rate, expressed as a weighted two-class problem so a
    # plain logistic regression can carry it: each objective contributes n*rate positive and
    # n*(1-rate) negative weight. This keeps well-measured objectives influential without
    # letting frequent ones dominate the vocabulary.
    texts = pd.concat([per_objective["text"], per_objective["text"]], ignore_index=True)
    labels = np.r_[np.ones(len(per_objective)), np.zeros(len(per_objective))]
    weights = np.r_[
        (per_objective["n"] * per_objective["rate"]).to_numpy(dtype=float),
        (per_objective["n"] * (1.0 - per_objective["rate"])).to_numpy(dtype=float),
    ]
    keep = weights > 0
    if int(labels[keep].sum()) == 0 or int((1 - labels[keep]).sum()) == 0:
        raise RuntimeError("lo_text_difficulty needs both outcomes present to fit")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min(min_df, max(1, len(per_objective) // 10)),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts[keep].fillna("").astype(str))
    classifier = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    classifier.fit(matrix, labels[keep], sample_weight=weights[keep])
    return LoTextDifficultyModel(vectorizer, classifier, fallback)


def _inner_oof(frame: pd.DataFrame, seed: int, **fit_kw: Any) -> np.ndarray:
    """Objective-grouped out-of-fold difficulty for the rows used to fit one outer model."""
    objectives = frame[LO_COL].astype(str).to_numpy()
    unique = np.unique(objectives)
    if len(unique) < 2 * INNER_SPLITS:
        # Too few objectives to hold any out — fall back to the training rate everywhere
        # rather than emitting a value that is quietly fitted on its own label.
        log.warning(
            "lo_text_difficulty: only %d objectives in this training fold; inner OOF "
            "disabled and the base rate used instead",
            len(unique),
        )
        return np.full(len(frame), float(frame[LABEL_COL].mean()), dtype=float)

    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_of = {obj: i % INNER_SPLITS for i, obj in enumerate(shuffled)}
    inner_fold = np.array([fold_of[o] for o in objectives])

    out = np.full(len(frame), np.nan, dtype=float)
    for k in range(INNER_SPLITS):
        tr = inner_fold != k
        va = ~tr
        if not tr.any() or not va.any():
            continue
        model = fit_lo_text_difficulty(frame.loc[tr], seed, **fit_kw)
        out[va] = model.predict(frame.loc[va, LO_TEXT_COL])
    if np.isnan(out).any():
        raise RuntimeError("inner lo_text_difficulty left rows unassigned; refusing to train")
    return out


def fold_safe_lo_text_difficulty(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seed: int,
    **fit_kw: Any,
) -> tuple[np.ndarray, LoTextDifficultyModel]:
    """Leakage-safe difficulty for one outer fold, plus the model to persist for inference.

    Returns a full-length vector: validation rows scored by the outer model, training rows by
    an inner objective-grouped OOF. The returned :class:`LoTextDifficultyModel` is the one
    fitted on the whole outer training fold — the one the submission must replay.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if train_mask.shape != (len(frame),) or valid_mask.shape != (len(frame),):
        raise ValueError("lo_text_difficulty masks must match the frame length")
    if np.any(train_mask & valid_mask):
        raise ValueError("lo_text_difficulty train and validation masks overlap")
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("lo_text_difficulty needs a non-empty train and validation side")

    outer = fit_lo_text_difficulty(frame.loc[train_mask], seed, **fit_kw)
    values = np.full(len(frame), np.nan, dtype=float)
    values[valid_mask] = outer.predict(frame.loc[valid_mask, LO_TEXT_COL])
    values[train_mask] = _inner_oof(frame.loc[train_mask], seed, **fit_kw)

    # Rows in neither mask exist when an objective fold purges them; they are not used by
    # this fold's model, but leaving NaN would silently propagate into LightGBM as "missing"
    # and look like a feature value rather than a bug.
    orphan = ~(train_mask | valid_mask)
    if orphan.any():
        values[orphan] = outer.fallback
    return values, outer
