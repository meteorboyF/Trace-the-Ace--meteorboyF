"""Deployment-oriented validation splits for objective and transcript-domain shift.

Ordinary session-grouped CV is necessary but not sufficient for Trace the Ace: it lets the
same learning objectives and transcript-generating domains occur on both sides. These split
tables are deliberately harsher. Objective folds purge every validation session from the
training side, while domain folds hold out clusters learned only from transcript features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from .config import get_config
from .io import LABEL_COL, load_train, write_parquet
from .logging_utils import get_logger
from .paths import interim_dir
from .tasks import task

log = get_logger("robust_cv")


def robust_folds_path(kind: str, fold_seed: int | None = None) -> Path:
    """Path for a robust fold table.

    ``fold_seed`` produces an *alternative assignment of the same objectives to folds*.
    Objective folds are far noisier than session folds — each holds only ~80 of 398
    objectives, and objective difficulty is lumpy — so a single assignment gives paired
    deltas with an SD around 0.037 AUC. Decisions need repeats (``evaluate.objective_repeated``),
    which is what these namespaced tables are for. ``None`` is the canonical assignment.
    """
    if kind not in {"objective", "domain"}:
        raise ValueError(f"unknown robust fold kind {kind!r}")
    suffix = "" if fold_seed is None else f"_s{int(fold_seed)}"
    return interim_dir() / f"folds_{kind}{suffix}.parquet"


def assign_objective_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Assign whole objectives to folds, balancing rows and positives greedily."""
    required = {"learning_objective_id", "session_id", "response_id", LABEL_COL}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"objective folds missing columns {sorted(missing)}")
    stats = (
        frame.groupby("learning_objective_id", as_index=False)
        .agg(n_rows=("response_id", "size"), n_pos=(LABEL_COL, "sum"))
        .sample(frac=1.0, random_state=seed)
        .sort_values("n_rows", ascending=False, kind="stable")
    )
    totals = np.zeros((n_splits, 2), dtype=float)
    assignment: dict[str, int] = {}
    target = np.array([len(frame) / n_splits, frame[LABEL_COL].sum() / n_splits])
    for index, row in enumerate(stats.itertuples(index=False)):
        contribution = np.array([float(row.n_rows), float(row.n_pos)])
        if index < n_splits:
            fold = index
        else:
            costs = []
            for k in range(n_splits):
                trial = totals.copy()
                trial[k] += contribution
                costs.append(float(np.var(trial / np.maximum(target, 1), axis=0).sum()))
            fold = int(np.argmin(costs))
        assignment[str(row.learning_objective_id)] = fold
        totals[fold] += contribution
    out = frame[["response_id", "session_id", "learning_objective_id", LABEL_COL]].copy()
    out["fold"] = out["learning_objective_id"].astype(str).map(assignment).astype(int)
    return out


def purged_split_indices(folds: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train/validation positions with both objective and session leakage purged."""
    validation = folds["fold"].eq(fold).to_numpy()
    valid_sessions = set(folds.loc[validation, "session_id"].astype(str))
    valid_objectives = set(folds.loc[validation, "learning_objective_id"].astype(str))
    training = (
        ~validation
        & ~folds["session_id"].astype(str).isin(valid_sessions).to_numpy()
        & ~folds["learning_objective_id"].astype(str).isin(valid_objectives).to_numpy()
    )
    return np.flatnonzero(training), np.flatnonzero(validation)


def _assign_domains_to_folds(
    frame: pd.DataFrame, domain_ids: pd.Series, n_splits: int
) -> pd.Series:
    work = frame.assign(domain_id=domain_ids.to_numpy())
    stats = (
        work.groupby("domain_id", as_index=False)
        .agg(n_rows=("response_id", "size"), n_pos=(LABEL_COL, "sum"))
        .sort_values("n_rows", ascending=False)
    )
    totals = np.zeros((n_splits, 2), dtype=float)
    target = np.array([len(work) / n_splits, work[LABEL_COL].sum() / n_splits])
    mapping: dict[int, int] = {}
    for index, row in enumerate(stats.itertuples(index=False)):
        add = np.array([float(row.n_rows), float(row.n_pos)])
        if index < n_splits:
            fold = index
        else:
            costs = []
            for k in range(n_splits):
                trial = totals.copy()
                trial[k] += add
                costs.append(float(np.var(trial / np.maximum(target, 1), axis=0).sum()))
            fold = int(np.argmin(costs))
        mapping[int(row.domain_id)] = fold
        totals[fold] += add
    return domain_ids.map(mapping).astype(int)


def _summary(frame: pd.DataFrame, kind: str, output: Path, cached: bool) -> dict[str, Any]:
    _validate_folds(frame, kind)
    per_fold: dict[int, dict[str, Any]] = {}
    for fold, valid in frame.groupby("fold"):
        entry: dict[str, Any] = {
            "n_rows": int(len(valid)),
            "n_sessions": int(valid["session_id"].nunique()),
            "n_objectives": int(valid["learning_objective_id"].nunique()),
            "pos_rate": round(float(valid[LABEL_COL].mean()), 4),
        }
        if kind == "objective":
            train_idx, _ = purged_split_indices(frame, int(fold))
            entry["n_train_after_purge"] = int(len(train_idx))
        per_fold[int(fold)] = entry
    return {
        "kind": kind,
        "output_path": str(output),
        "n_rows": int(len(frame)),
        "n_folds": int(frame["fold"].nunique()),
        "per_fold": per_fold,
        "cached": cached,
    }


def _validate_folds(frame: pd.DataFrame, kind: str) -> None:
    if frame["fold"].nunique() != 5 or frame["fold"].isna().any():
        raise RuntimeError(f"{kind} folds must contain exactly five complete assignments")
    if kind == "domain" and frame.groupby("session_id")["fold"].nunique().max() != 1:
        raise RuntimeError("domain folds leak a session across validation folds")
    if kind == "objective":
        if frame.groupby("learning_objective_id")["fold"].nunique().max() != 1:
            raise RuntimeError("objective folds split a learning objective")
        objective_counts = frame.groupby("fold")["learning_objective_id"].nunique()
        if frame["learning_objective_id"].nunique() >= 100 and int(objective_counts.min()) < 20:
            raise RuntimeError(
                "objective folds are degenerate: each full-data fold must hold at least "
                f"20 objectives, got {objective_counts.to_dict()}"
            )
        for fold in sorted(frame["fold"].unique()):
            train_idx, valid_idx = purged_split_indices(frame, int(fold))
            if not len(train_idx) or not len(valid_idx):
                raise RuntimeError(f"objective fold {fold} is empty after purging")


@task(
    "cv.robust_build",
    requires="cpu",
    max_tier="cpu",
    description="build purged objective-disjoint and transcript-domain validation folds",
)
def build(
    force: bool = False,
    subsample: int | None = None,
    kind: str = "objective",
    n_domains: int = 20,
    fold_seed: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    output = robust_folds_path(kind, fold_seed)
    if output.is_file() and not force and subsample is None:
        return _summary(pd.read_parquet(output), kind, output, cached=True)
    frame = load_train()
    if subsample is not None:
        sessions = frame["session_id"].drop_duplicates().head(subsample)
        frame = frame[frame["session_id"].isin(sessions)].copy()
    n_splits = int(cfg.cv["n_splits"])
    if kind == "objective":
        folds = assign_objective_folds(
            frame, n_splits, int(cfg.seed if fold_seed is None else fold_seed)
        )
    elif kind == "domain":
        from .features.assemble import build_matrix

        base = frame[["response_id", "session_id"]].copy()
        matrix, feature_cols = build_matrix(base, blocks=["structural", "linguistic", "temporal"])
        session_matrix = matrix[["session_id", *feature_cols]].drop_duplicates("session_id")
        values = session_matrix[feature_cols].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median()).to_numpy(dtype=np.float32)
        values = RobustScaler(quantile_range=(10, 90)).fit_transform(values)
        clusters = MiniBatchKMeans(
            n_clusters=min(n_domains, len(session_matrix)),
            random_state=int(cfg.seed),
            batch_size=2048,
            n_init=10,
        ).fit_predict(values)
        domain_by_session = dict(zip(session_matrix["session_id"].astype(str), clusters))
        domain_ids = frame["session_id"].astype(str).map(domain_by_session).astype(int)
        folds = frame[["response_id", "session_id", "learning_objective_id", LABEL_COL]].copy()
        folds["domain_id"] = domain_ids
        folds["fold"] = _assign_domains_to_folds(folds, domain_ids, n_splits)
    else:
        raise ValueError("kind must be 'objective' or 'domain'")
    if subsample is not None:
        suffix = "" if fold_seed is None else f"_s{int(fold_seed)}"
        output = interim_dir() / f"folds_{kind}_sub{subsample}{suffix}.parquet"
    _validate_folds(folds, kind)
    write_parquet(folds, output)
    log.info("cv.robust_build: %s, %d rows -> %s", kind, len(folds), output)
    return _summary(folds, kind, output, cached=False)


def load_robust_folds(
    kind: str, subsample: int | None = None, fold_seed: int | None = None
) -> pd.DataFrame:
    if subsample is None:
        output = robust_folds_path(kind, fold_seed)
    else:
        suffix = "" if fold_seed is None else f"_s{int(fold_seed)}"
        output = interim_dir() / f"folds_{kind}_sub{subsample}{suffix}.parquet"
    if not output.is_file():
        raise FileNotFoundError(
            f"{output} missing — run cv.robust_build(kind={kind!r}"
            + (f", fold_seed={fold_seed}" if fold_seed is not None else "")
            + ")"
        )
    return pd.read_parquet(output)
