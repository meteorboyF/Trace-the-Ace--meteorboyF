"""Unseen-learning-objective stress test.

**Why this exists.** The organizers confirmed on the forum (2026-07-06, `kwetstone`) that
*"not every learning objective in the test set appears in the training set."* Our
session-grouped CV barely exercises that regime — only **0.27%** of validation rows have an
objective absent from their training fold — so the headline CV score is measured almost
entirely in the *seen-objective* regime and is optimistic to the extent the test set differs.

This matters more than it sounds, because ``lo_prior_enc`` (the per-objective difficulty
lookup) is the model's highest-gain feature. On an unseen objective it degrades to the global
base rate, and the ``baseline.lo_only`` bar degrades to a *constant* — on the 95 natural
unseen rows it scored 0.6154, **worse than predicting the global prior**. Whatever accuracy
survives there comes from the transcript, which is precisely the signal the organizers say
strong submissions should rely on.

**Design.** We deliberately hold out whole objectives, while preserving the session-grouping
invariant that makes any of this valid:

1. Sample a fraction of ``learning_objective_id`` values to hold out.
2. Move **entire sessions** to validation if they contain *any* held-out objective. This is
   the crucial step: assigning rows individually would put one session's transcript on both
   sides of the split and leak it.
3. Train on the remaining sessions, then score **only** the validation rows whose objective is
   genuinely absent from the training side.

The result is an honest estimate of the regime the test set will partly be in, and a direct
measurement of how much the transcript features carry when the topic prior cannot help.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import logloss
from .io import LABEL_COL, load_train_features
from .logging_utils import get_logger
from .paths import runs_dir
from .progress import pbar
from .repeated import summarize
from .tasks import task

log = get_logger("unseen_lo")

LO_COL = "learning_objective_id"


def _fit_weight_with_bases(y: np.ndarray, predictions: np.ndarray, base_rates: np.ndarray) -> float:
    """Fit one shrinkage weight when proxy draws have different train-only base rates."""
    grid = np.arange(0.05, 1.01, 0.05)
    losses = [
        logloss(y, np.clip(base_rates + w * (predictions - base_rates), 1e-6, 1 - 1e-6))
        for w in grid
    ]
    return float(grid[int(np.argmin(losses))])


def _crossfit_shrinkage(
    draws: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate shrinkage without fitting a row's weight or base rate on its label.

    Each tuple is ``(response_ids, labels, predictions, train_base_rate)``. Proxy draws
    overlap, so response IDs in the evaluation draw are explicitly removed from the other
    draws used to fit its weight.
    """
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for eval_i, (eval_ids, eval_y, eval_p, eval_base) in enumerate(draws):
        fit_y: list[np.ndarray] = []
        fit_p: list[np.ndarray] = []
        fit_b: list[np.ndarray] = []
        eval_id_set = set(eval_ids.tolist())
        for fit_i, (ids, y, p, base) in enumerate(draws):
            if fit_i == eval_i:
                continue
            keep = np.asarray([value not in eval_id_set for value in ids], dtype=bool)
            if keep.any():
                fit_y.append(y[keep])
                fit_p.append(p[keep])
                fit_b.append(np.full(int(keep.sum()), base))
        if not fit_y:
            raise RuntimeError("shrinkage cross-fit has no non-overlapping calibration rows")
        weight = _fit_weight_with_bases(
            np.concatenate(fit_y), np.concatenate(fit_p), np.concatenate(fit_b)
        )
        ys.append(eval_y)
        preds.append(np.clip(eval_base + weight * (eval_p - eval_base), 1e-6, 1 - 1e-6))
    return np.concatenate(ys), np.concatenate(preds)


def _split(
    df: pd.DataFrame, holdout_frac: float, rng: np.random.Generator
) -> tuple[pd.Index, pd.Index, set[str]]:
    """Return (train_idx, val_idx, held_out_los) with sessions kept intact."""
    los = df[LO_COL].dropna().unique()
    n_hold = max(1, int(round(len(los) * holdout_frac)))
    held = set(rng.choice(los, size=n_hold, replace=False).tolist())

    # any session touching a held-out objective goes ENTIRELY to validation
    touched = set(df.loc[df[LO_COL].isin(held), "session_id"])
    val_mask = df["session_id"].isin(touched)
    return df.index[~val_mask], df.index[val_mask], held


@task(
    "evaluate.unseen_lo",
    requires="cpu",
    max_tier="cpu",
    description="stress test on learning objectives absent from training (the real test regime)",
)
def stress_test(
    force: bool = False,
    subsample: int | None = None,
    holdout_frac: float = 0.25,
    n_repeats: int = 5,
    blocks: list[str] | None = None,
    num_boost_round: int = 800,
    seed: int | None = None,
    lo_smoothing: float = 20.0,
) -> dict[str, Any]:
    """Measure performance on genuinely unseen learning objectives.

    Reports, with error bars over ``n_repeats`` different hold-out draws:
    the transcript model, the LO-only baseline, and the global prior — each restricted to
    validation rows whose objective never appeared in training.
    """
    import lightgbm as lgb

    from .config import get_config
    from .features.assemble import DEFAULT_BLOCKS, build_matrix
    from .models.gbdt import (
        DEFAULT_PARAMS,
        LO_ENC_COL,
        _inner_oof_lo_encoding,
        _smoothed_map,
    )

    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    blocks = list(blocks or DEFAULT_BLOCKS)

    feats = load_train_features()
    labels = pd.read_csv(cfg.work_dir / "data" / "raw" / cfg.canonical_files["train_labels"])
    labels = labels.rename(columns={"is_correct": LABEL_COL})
    labels["response_id"] = labels["response_id"].astype(str)
    base = feats.merge(labels[["response_id", LABEL_COL]], on="response_id", how="inner")
    if subsample is not None:
        sess = base["session_id"].drop_duplicates().head(max(1, subsample))
        base = base[base["session_id"].isin(sess)]

    frame, feat_cols = build_matrix(base.reset_index(drop=True), blocks=blocks, subsample=subsample)
    frame[LABEL_COL] = frame[LABEL_COL].astype(float)

    model_ll: list[float] = []
    prior_ll: list[float] = []
    n_unseen: list[float] = []

    for r in pbar(range(n_repeats), desc="unseen-LO repeats", unit="draw"):
        rng = np.random.default_rng(seed + r)
        tr_idx, va_idx, held = _split(frame, holdout_frac, rng)
        tr, va = frame.loc[tr_idx], frame.loc[va_idx]
        if len(tr) < 100 or len(va) < 50:
            continue

        # score ONLY rows whose objective is truly absent from training
        seen_los = set(tr[LO_COL])
        va_unseen = va[~va[LO_COL].isin(seen_los)]
        if len(va_unseen) < 30:
            continue

        p = dict(DEFAULT_PARAMS)
        p.update({"seed": seed + r, "bagging_seed": seed + r, "feature_fraction_seed": seed + r})
        model_cols = [*feat_cols, LO_ENC_COL]
        x_train = tr[feat_cols].copy()
        x_valid = va_unseen[feat_cols].copy()
        x_train[LO_ENC_COL] = _inner_oof_lo_encoding(tr, smoothing=lo_smoothing, seed=seed + r)
        lo_map, global_rate = _smoothed_map(tr, smoothing=lo_smoothing)
        x_valid[LO_ENC_COL] = va_unseen[LO_COL].map(lo_map).fillna(global_rate).to_numpy()
        booster = lgb.train(
            p,
            lgb.Dataset(x_train[model_cols], label=tr[LABEL_COL].to_numpy()),
            num_boost_round=num_boost_round,
        )
        y = va_unseen[LABEL_COL].to_numpy()
        pred = np.asarray(booster.predict(x_valid[model_cols]), dtype=float)

        model_ll.append(logloss(y, pred))
        prior_ll.append(logloss(y, np.full(len(y), global_rate)))
        # NOTE: lo_only on an unseen objective has nothing to look up, so it degenerates
        # to exactly the global rate — the same number as `prior`. They coincide here.
        n_unseen.append(float(len(va_unseen)))

    if not model_ll:
        raise RuntimeError("no usable hold-out draws — try a larger holdout_frac or subsample")

    seeds = list(range(len(model_ll)))
    res_model = summarize("model (unseen LO)", model_ll, seeds)
    res_prior = summarize("prior/lo_only (unseen LO)", prior_ll, seeds)
    res_n = summarize("n rows scored", n_unseen, seeds)
    gain = summarize(
        "transcript gain on unseen LO", [m - b for m, b in zip(model_ll, prior_ll)], seeds
    )

    out: dict[str, Any] = {
        "holdout_frac": holdout_frac,
        "n_repeats": len(model_ll),
        "blocks": blocks,
        "include_lo_prior": True,
        "lo_smoothing": lo_smoothing,
        "num_boost_round": num_boost_round,
        "model_unseen": res_model.to_dict(),
        "prior_unseen": res_prior.to_dict(),
        "rows_scored": res_n.to_dict(),
        "transcript_gain_unseen": gain.to_dict(),
        "logloss": res_model.mean,
    }
    d = runs_dir() / "unseen_lo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stress.json").write_text(json.dumps(out, indent=2, default=str))
    out["output_path"] = str(d / "stress.json")

    print(
        f"\nUNSEEN-LEARNING-OBJECTIVE STRESS TEST ({len(model_ll)} draws, "
        f"{holdout_frac:.0%} of objectives held out)"
    )
    print(f"  rows scored per draw : {res_n.mean:.0f} ± {res_n.sd:.0f}")
    print(f"  model                : {res_model.mean:.5f} ± {res_model.sd:.5f}")
    print(f"  prior == lo_only     : {res_prior.mean:.5f} ± {res_prior.sd:.5f}")
    print(
        f"  transcript gain      : {gain.mean:+.5f} ± {gain.sd:.5f}  "
        f"({'REAL' if gain.significant else 'not distinguishable from 0'})"
    )
    return out


@task(
    "calibrate.shrinkage",
    requires="cpu",
    max_tier="cpu",
    description="fit deployment shrinkage on the unseen-objective regime (test-like conditions)",
)
def fit_deployment_shrinkage(
    force: bool = False,
    subsample: int | None = None,
    holdout_frac: float = 0.25,
    n_repeats: int = 4,
    blocks: list[str] | None = None,
    num_boost_round: int = 800,
    seed: int | None = None,
    lo_smoothing: float = 20.0,
    experiment: str = "model.gbdt",
) -> dict[str, Any]:
    """Fit the shrinkage weight that corrects deployment overconfidence.

    **Why this is necessary.** Our CV regime is easier than the test regime: 99.7% of CV
    validation rows have a learning objective the model has seen, and every session comes
    from one provider. The organizers confirmed the test set contains unseen objectives and
    is not drawn entirely from one source. A model tuned on the easier regime is
    systematically overconfident when deployed — the leaderboard showed exactly that
    signature (AUROC 0.6014, i.e. real ranking signal, with log loss 0.6106, marginally
    *worse* than the global base rate).

    Shrinkage cannot change ranking (AUC is invariant under it); it only pulls the
    magnitudes toward the base rate. The weight is fitted **here, on training data** held
    out by objective — never against leaderboard feedback, which would be rules-adjacent.
    """
    import joblib
    import lightgbm as lgb

    from .calibration import fit_shrinkage
    from .config import get_config
    from .evaluate import auc, experiment_dir
    from .features.assemble import DEFAULT_BLOCKS, build_matrix
    from .models.gbdt import (
        DEFAULT_PARAMS,
        LO_ENC_COL,
        _inner_oof_lo_encoding,
        _smoothed_map,
    )

    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    blocks = list(blocks or DEFAULT_BLOCKS)

    feats = load_train_features()
    labels = pd.read_csv(cfg.work_dir / "data" / "raw" / cfg.canonical_files["train_labels"])
    labels = labels.rename(columns={"is_correct": LABEL_COL})
    labels["response_id"] = labels["response_id"].astype(str)
    base_df = feats.merge(labels[["response_id", LABEL_COL]], on="response_id", how="inner")
    if subsample is not None:
        sess = base_df["session_id"].drop_duplicates().head(max(1, subsample))
        base_df = base_df[base_df["session_id"].isin(sess)]

    frame, feat_cols = build_matrix(base_df.reset_index(drop=True), blocks=blocks)
    frame[LABEL_COL] = frame[LABEL_COL].astype(float)
    base_rate = float(frame[LABEL_COL].mean())

    draws: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for r in pbar(range(n_repeats), desc="calibrate.shrinkage draws", unit="draw"):
        rng = np.random.default_rng(seed + r)
        tr_idx, va_idx, _ = _split(frame, holdout_frac, rng)
        tr, va = frame.loc[tr_idx], frame.loc[va_idx]
        va_unseen = va[~va[LO_COL].isin(set(tr[LO_COL]))]
        if len(tr) < 100 or len(va_unseen) < 30:
            continue
        p = dict(DEFAULT_PARAMS)
        p.update({"seed": seed + r, "bagging_seed": seed + r, "feature_fraction_seed": seed + r})
        cols = [*feat_cols, LO_ENC_COL]
        x_tr, x_va = tr[feat_cols].copy(), va_unseen[feat_cols].copy()
        x_tr[LO_ENC_COL] = _inner_oof_lo_encoding(tr, smoothing=lo_smoothing, seed=seed + r)
        lo_map, global_rate = _smoothed_map(tr, smoothing=lo_smoothing)
        x_va[LO_ENC_COL] = va_unseen[LO_COL].map(lo_map).fillna(global_rate).to_numpy()
        bst = lgb.train(
            p,
            lgb.Dataset(x_tr[cols], label=tr[LABEL_COL].to_numpy()),
            num_boost_round=num_boost_round,
        )
        draws.append(
            (
                va_unseen["response_id"].astype(str).to_numpy(),
                va_unseen[LABEL_COL].to_numpy(),
                np.asarray(bst.predict(x_va[cols]), dtype=float),
                float(global_rate),
            )
        )

    if not draws:
        raise RuntimeError("no usable hold-out draws for shrinkage fitting")

    y = np.concatenate([draw[1] for draw in draws])
    p_all = np.concatenate([draw[2] for draw in draws])
    weight = fit_shrinkage(y, p_all, base_rate)

    before = logloss(y, p_all)
    fitted_after = logloss(y, np.clip(base_rate + weight * (p_all - base_rate), 1e-6, 1 - 1e-6))
    crossfit_y, crossfit_p = _crossfit_shrinkage(draws)
    after = logloss(crossfit_y, crossfit_p)
    prior = logloss(y, np.full(len(y), base_rate))

    out: dict[str, Any] = {
        "weight": float(weight),
        "base_rate": base_rate,
        "n_holdout_rows": int(len(y)),
        "logloss_unshrunk": before,
        "logloss_shrunk": after,
        "logloss_shrunk_in_sample": fitted_after,
        "logloss_constant_prior": prior,
        "gain": before - after,
        "gain_in_sample": before - fitted_after,
        "auc_unchanged": auc(y, p_all),
        "logloss": after,
    }
    dest = experiment_dir(experiment, None)
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump({"weight": float(weight), "base_rate": base_rate}, dest / "shrinkage.joblib")
    out["output_path"] = str(dest / "shrinkage.joblib")

    log.info(
        "shrinkage w=%.2f: %.5f -> %.5f (gain %+.5f; constant prior %.5f)",
        weight,
        before,
        after,
        before - after,
        prior,
    )
    return out
