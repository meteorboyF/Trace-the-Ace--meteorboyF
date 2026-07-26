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
from ..evaluate import experiment_dir, save_oof, score_frame
from ..features.assemble import DEFAULT_BLOCKS, build_matrix, summarize
from ..io import LABEL_COL, load_train_features, write_parquet
from ..logging_utils import get_logger
from ..paths import runs_dir
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


def _inner_oof_lo_encoding(frame: pd.DataFrame, smoothing: float, seed: int) -> np.ndarray:
    """Session-grouped OOF target encoding for rows used to fit one model."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = int(frame["session_id"].nunique())
    n_inner = min(5, n_groups)
    if n_inner < 2:
        raise ValueError(
            f"only {n_groups} training session(s); cannot target-encode without self-label leakage"
        )
    splitter = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    y = frame[LABEL_COL].to_numpy()
    groups = frame["session_id"].to_numpy()
    try:
        splits = list(splitter.split(frame, y, groups))
    except ValueError as exc:
        raise ValueError("cannot build leakage-safe inner target-encoding folds") from exc

    encoded = np.full(len(frame), np.nan, dtype=float)
    for inner_tr_idx, inner_va_idx in splits:
        inner_map, inner_global = _smoothed_map(frame.iloc[inner_tr_idx], smoothing)
        encoded[inner_va_idx] = (
            frame.iloc[inner_va_idx][LO_COL].map(inner_map).fillna(inner_global).to_numpy()
        )
    if np.isnan(encoded).any():
        raise RuntimeError("inner target encoding left rows unassigned; refusing to train")
    return encoded


def _fold_safe_lo_encoding(
    frame: pd.DataFrame, smoothing: float, seed: int, outer_fold: int
) -> np.ndarray:
    """Target-encode the learning objective for one outer-fold model.

    Two levels of protection:

    * **Validation rows** of ``outer_fold`` are encoded from that fold's training rows only,
      so the OOF estimate never sees its own labels.
    * **Training rows** are encoded with an *inner* session-grouped K-fold, so the model does
      not learn to trust an encoding that was computed from the very row it is predicting.
      Skipping this inner loop makes the encoding look far more reliable during training than
      it is at validation time, and the model over-relies on it.

    The encoding must be built separately for every outer model. A single shared vector
    cannot represent this quantity: every row is training data for four outer folds and
    validation data for one, so its correct value differs by outer fold.
    """
    enc = np.full(len(frame), np.nan, dtype=float)
    known_folds = {int(k) for k in frame["fold"].unique()}
    if outer_fold not in known_folds:
        raise ValueError(
            f"outer fold {outer_fold} is absent from frame folds {sorted(known_folds)}"
        )

    tr_mask = (frame["fold"] != outer_fold).to_numpy()
    va_mask = (frame["fold"] == outer_fold).to_numpy()
    tr = frame.loc[tr_mask]
    if tr.empty or not va_mask.any():
        raise ValueError(f"outer fold {outer_fold} has an empty train or validation partition")

    # --- validation rows: encode from the whole OUTER training fold ------------
    smoothed, global_rate = _smoothed_map(tr, smoothing)
    enc[va_mask] = frame.loc[va_mask, LO_COL].map(smoothed).fillna(global_rate).to_numpy()

    # --- training rows: inner out-of-fold encoding ------------------------------
    enc[tr_mask] = _inner_oof_lo_encoding(tr, smoothing, seed)

    if np.isnan(enc).any():
        raise RuntimeError(
            f"target encoding for outer fold {outer_fold} left rows unassigned; refusing to train"
        )
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
    cv_seed: int | None = None,
) -> dict[str, Any]:
    import lightgbm as lgb

    from ..config import get_config

    cfg = get_config()
    seed = int(seed if seed is not None else cfg.seed)
    blocks = list(blocks or DEFAULT_BLOCKS)

    folds = load_folds(subsample=subsample, cv_seed=cv_seed)
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
    include_lo_prior = bool(include_lo_prior and LO_COL in frame.columns)
    base_feat_cols = list(feat_cols)
    if include_lo_prior:
        feat_cols = [*feat_cols, LO_ENC_COL]

    feature_summary = summarize(frame, base_feat_cols)
    if include_lo_prior:
        feature_summary["n_features"] += 1
        by_block = feature_summary["features_by_block"]
        by_block["lo_prior"] = by_block.get("lo_prior", 0) + 1
    log.info("gbdt: %s", feature_summary)

    X_base = frame[base_feat_cols]
    y = frame[LABEL_COL].to_numpy(dtype=float)
    oof = np.full(len(frame), np.nan, dtype=float)

    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    p["seed"] = seed
    p["bagging_seed"] = seed
    p["feature_fraction_seed"] = seed

    importances: list[pd.Series] = []
    best_iters: list[int] = []
    boosters: list[Any] = []
    fold_lo_priors: list[dict[str, Any]] = []
    fold_ids = sorted(frame["fold"].unique())

    for k in pbar(fold_ids, desc="model.gbdt folds", unit="fold"):
        tr = frame["fold"] != k
        va = frame["fold"] == k
        if include_lo_prior:
            X = X_base.copy()
            X[LO_ENC_COL] = _fold_safe_lo_encoding(frame, lo_smoothing, seed, outer_fold=int(k))
            outer_map, outer_global = _smoothed_map(frame.loc[tr], lo_smoothing)
            fold_lo_priors.append(
                {
                    "fold": int(k),
                    "fallback": float(outer_global),
                    "values": {str(key): float(value) for key, value in outer_map.items()},
                }
            )
        else:
            X = X_base
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
        boosters.append(booster)
        best_iters.append(int(booster.best_iteration or num_boost_round))
        importances.append(
            pd.Series(booster.feature_importance("gain"), index=feat_cols, dtype=float)
        )

    frame["pred"] = oof
    oof_cols = ["response_id", "session_id", LABEL_COL, "pred", "learning_objective_id"]
    # carry slice columns onto the OOF frame so evaluate.report can slice without re-joining
    for c in ("struct_n_utterances", "struct_student_talk_ratio"):
        if c in frame.columns:
            oof_cols.append(c)
    save_oof(experiment, frame[oof_cols], subsample=subsample, cv_seed=cv_seed)

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
    mdir = experiment_dir(experiment, subsample, cv_seed)
    mdir.mkdir(parents=True, exist_ok=True)
    temp_paths: list[tuple[Any, Any]] = []
    try:
        # Write a complete new generation before touching the prior one. A failed fit or
        # write therefore cannot leave a plausible mix of old and new fold models.
        for k, booster in zip(fold_ids, boosters):
            tmp = mdir / f".fold{k}.txt.tmp"
            booster.save_model(str(tmp), num_iteration=booster.best_iteration)
            temp_paths.append((tmp, mdir / f"fold{k}.txt"))
        imp_tmp = mdir / ".importance.parquet.tmp"
        write_parquet(imp_summary, imp_tmp)
        temp_paths.append((imp_tmp, mdir / "importance.parquet"))

        order_tmp = mdir / ".feature_order.json.tmp"
        order_tmp.write_text(json.dumps(list(feat_cols), indent=0))
        temp_paths.append((order_tmp, mdir / "feature_order.json"))

        for spec in fold_lo_priors:
            k = int(spec["fold"])
            tmp = mdir / f".lo_prior_fold{k}.json.tmp"
            tmp.write_text(json.dumps(spec, sort_keys=True))
            temp_paths.append((tmp, mdir / f"lo_prior_fold{k}.json"))

        model_manifest = {
            "experiment": experiment,
            "subsample": subsample,
            "cv_seed": cv_seed,
            "fold_ids": [int(k) for k in fold_ids],
            "feature_cols": feat_cols,
            "blocks": blocks,
            "params": p,
            "seed": seed,
            "include_lo_prior": include_lo_prior,
            "lo_smoothing": lo_smoothing,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
        }
        manifest_tmp = mdir / ".training_manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(model_manifest, indent=2, default=str))
        temp_paths.append((manifest_tmp, mdir / "training_manifest.json"))

        # Invalidate every dependent artifact from the previous generation, including
        # calibration fitted to different OOF predictions.
        for old in [
            *mdir.glob("fold*.txt"),
            *mdir.glob("lo_prior_fold*.json"),
            mdir / "importance.parquet",
            mdir / "feature_order.json",
            mdir / "training_manifest.json",
            mdir / "calibrator.joblib",
            mdir / "calibrator_plain.joblib",
        ]:
            old.unlink(missing_ok=True)
        for tmp, final in temp_paths:
            tmp.replace(final)
    except Exception:
        for tmp, _ in temp_paths:
            tmp.unlink(missing_ok=True)
        raise

    imp_path = mdir / "importance.parquet"

    res = score_frame(frame, experiment, subsample=subsample, cv_seed=cv_seed)
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
