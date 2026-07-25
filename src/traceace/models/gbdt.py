"""LightGBM on the handcrafted feature blocks.

CPU-only by design: at ~35k rows x a few hundred features, LightGBM on 24 vCPUs is
faster than the GPU histogram build and costs zero units. The tier guard enforces this
(``max_tier="cpu"``) so this never silently burns an attached A100.

Trains one model per fold on the persisted session-grouped folds, writes OOF predictions
(required for calibration and blending, §10.2), and reports log loss with the delta
against ``baseline.lo_only`` — the organizers' anti-goal bar.

Per-fold models and the cross-fold feature importances are persisted so
``interpret.report`` can show importance with confidence intervals rather than a
single-fit bar chart.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..cv import load_folds
from ..evaluate import save_oof, score_frame
from ..features.assemble import DEFAULT_BLOCKS, build_matrix, summarize
from ..io import LABEL_COL, load_train_features, write_parquet
from ..logging_utils import get_logger
from ..paths import models_dir, runs_dir
from ..progress import heartbeat, pbar
from ..tasks import task

log = get_logger("model.gbdt")

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,  # all cores
    "force_col_wise": True,
}


LO_COL = "learning_objective_id"
LO_ENC_COL = "lo_prior_enc"


def _smoothed_map(df: pd.DataFrame, smoothing: float) -> tuple[pd.Series, float]:
    """Empirical-Bayes smoothed per-LO correctness from ``df``, plus the global rate."""
    global_rate = float(df[LABEL_COL].mean())
    agg = df.groupby(LO_COL)[LABEL_COL].agg(["sum", "count"])
    smoothed = (agg["sum"] + smoothing * global_rate) / (agg["count"] + smoothing)
    return smoothed, global_rate


def _fold_safe_lo_encoding(frame: pd.DataFrame, smoothing: float, seed: int) -> np.ndarray:
    """Target-encode the learning objective without leaking labels.

    Two levels of protection:

    * **Validation rows** of outer fold *k* are encoded from fold *k*'s training rows only,
      so the OOF estimate never sees its own labels.
    * **Training rows** are encoded with an *inner* session-grouped K-fold, so the model does
      not learn to trust an encoding that was computed from the very row it is predicting.
      Skipping this inner loop makes the encoding look far more reliable during training than
      it is at validation time, and the model over-relies on it.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    enc = np.full(len(frame), np.nan, dtype=float)
    for k in sorted(frame["fold"].unique()):
        tr_mask = (frame["fold"] != k).to_numpy()
        va_mask = (frame["fold"] == k).to_numpy()
        tr = frame.loc[tr_mask]

        # --- validation rows: encode from the whole training fold ---------------
        smoothed, global_rate = _smoothed_map(tr, smoothing)
        enc[va_mask] = frame.loc[va_mask, LO_COL].map(smoothed).fillna(global_rate).to_numpy()

        # --- training rows: inner out-of-fold encoding --------------------------
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        tr_y = tr[LABEL_COL].to_numpy()
        tr_groups = tr["session_id"].to_numpy()
        tr_positions = np.where(tr_mask)[0]
        try:
            splits = list(inner.split(tr, tr_y, tr_groups))
        except ValueError:  # too few groups (tiny subsample) — fall back
            splits = []
        if not splits:
            enc[tr_mask] = tr[LO_COL].map(smoothed).fillna(global_rate).to_numpy()
            continue
        for inner_tr_idx, inner_va_idx in splits:
            inner_map, inner_global = _smoothed_map(tr.iloc[inner_tr_idx], smoothing)
            target = tr_positions[inner_va_idx]
            enc[target] = (
                tr.iloc[inner_va_idx][LO_COL].map(inner_map).fillna(inner_global).to_numpy()
            )

    if np.isnan(enc).any():  # safety net; should not happen
        enc = np.nan_to_num(enc, nan=float(frame[LABEL_COL].mean()))
    return enc


@task(
    "model.gbdt",
    requires="cpu",
    max_tier="cpu",
    description="LightGBM over handcrafted feature blocks (CPU; beats GPU at this size)",
)
def train(
    force: bool = False,
    subsample: int | None = None,
    blocks: list[str] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    experiment: str = "model.gbdt",
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    include_lo_prior: bool = True,
    lo_smoothing: float = 20.0,
) -> dict[str, Any]:
    import lightgbm as lgb

    from ..config import get_config

    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    blocks = list(blocks or DEFAULT_BLOCKS)

    folds = load_folds(subsample=subsample)
    feats = load_train_features()
    base = folds.merge(
        feats[["response_id", "session_id", "learning_objective_id"]],
        on=["response_id", "session_id"],
        how="left",
    )
    frame, feat_cols = build_matrix(base, blocks=blocks, subsample=subsample)
    if not feat_cols:
        raise RuntimeError("no numeric features assembled — did the feature tasks run?")

    # Learning-objective difficulty as an explicit feature.
    #
    # Without it the model competes AGAINST topic difficulty instead of adding to it:
    # measured, transcript-only features score 0.593 against a 0.552 lo_only bar. The
    # organizers' anti-goal is a model that uses ONLY this signal — not one that refuses
    # to use it — so we include it and keep reporting delta_vs_lo_only, which now measures
    # exactly what the transcript contributes ON TOP of knowing the topic.
    if include_lo_prior and LO_COL in frame.columns:
        frame[LO_ENC_COL] = _fold_safe_lo_encoding(frame, lo_smoothing, seed)
        feat_cols = [*feat_cols, LO_ENC_COL]

    log.info("gbdt: %s", summarize(frame, feat_cols))

    X = frame[feat_cols]
    y = frame[LABEL_COL].to_numpy(dtype=float)
    oof = np.full(len(frame), np.nan, dtype=float)

    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    p["seed"] = seed
    p["bagging_seed"] = seed
    p["feature_fraction_seed"] = seed

    importances: list[pd.Series] = []
    best_iters: list[int] = []
    fold_ids = sorted(frame["fold"].unique())

    for k in pbar(fold_ids, desc="model.gbdt folds", unit="fold"):
        tr = frame["fold"] != k
        va = frame["fold"] == k
        dtrain = lgb.Dataset(X[tr], label=y[tr.to_numpy()], free_raw_data=False)
        dvalid = lgb.Dataset(X[va], label=y[va.to_numpy()], reference=dtrain, free_raw_data=False)

        with heartbeat(f"lgb fit fold {k}"):
            booster = lgb.train(
                p,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    # native streaming progress; silent when bars are disabled
                    lgb.log_evaluation(period=200),
                ],
            )
        oof[va.to_numpy()] = booster.predict(X[va], num_iteration=booster.best_iteration)
        best_iters.append(int(booster.best_iteration or num_boost_round))
        importances.append(
            pd.Series(booster.feature_importance("gain"), index=feat_cols, dtype=float)
        )
        mdir = models_dir() / experiment.replace(".", "_")
        mdir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(mdir / f"fold{k}.txt"), num_iteration=booster.best_iteration)

    frame["pred"] = oof
    oof_cols = ["response_id", "session_id", LABEL_COL, "pred", "learning_objective_id"]
    # carry slice columns onto the OOF frame so evaluate.report can slice without re-joining
    for c in ("struct_n_utterances", "struct_student_talk_ratio"):
        if c in frame.columns:
            oof_cols.append(c)
    save_oof(experiment, frame[oof_cols])

    # cross-fold importance with dispersion (for interpret.report)
    imp = pd.concat(importances, axis=1)
    imp.columns = [f"fold{k}" for k in fold_ids]
    imp_summary = pd.DataFrame(
        {
            "feature": imp.index,
            "gain_mean": imp.mean(axis=1).to_numpy(),
            "gain_std": imp.std(axis=1).to_numpy(),
            "gain_min": imp.min(axis=1).to_numpy(),
            "gain_max": imp.max(axis=1).to_numpy(),
        }
    ).sort_values("gain_mean", ascending=False)
    imp_path = models_dir() / experiment.replace(".", "_") / "importance.parquet"
    write_parquet(imp_summary, imp_path)

    res = score_frame(frame, experiment)
    res.update(
        {
            "blocks": blocks,
            "n_features": len(feat_cols),
            "best_iterations": best_iters,
            "importance_path": str(imp_path),
            "top_features": imp_summary.head(15)["feature"].tolist(),
        }
    )
    d = runs_dir() / "gbdt"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{experiment.replace('.', '_')}.json").write_text(json.dumps(res, indent=2, default=str))

    delta = res.get("delta_vs_lo_only")
    log.info(
        "gbdt %s: logloss=%.5f auc=%.4f%s",
        experiment,
        res["logloss"],
        res["auc"],
        f" delta_vs_lo_only={delta:+.5f}" if delta is not None else "",
    )
    return res
