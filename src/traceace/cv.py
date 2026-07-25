"""Cross-validation folds — grouped by session, never by response.

**The single most important correctness invariant in this repo.** One tutoring
session yields up to 10 response rows (different learning objectives) that share an
*identical transcript*. Splitting by ``response_id`` would put the same transcript in
both train and validation, leaking it, and every score would be fiction.

We use ``StratifiedGroupKFold(groups=session_id, y=correct)`` so folds are
session-disjoint while keeping the label balance even. Folds are generated **once**,
persisted to ``data/interim/folds.parquet``, and reused by every model — so all
experiments are comparable and OOF predictions can be blended.

``tests/test_cv_leakage.py`` asserts zero session overlap and runs in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import get_config
from .io import LABEL_COL, load_train, write_parquet
from .logging_utils import get_logger
from .paths import interim_dir
from .tasks import task

log = get_logger("cv")


def folds_path() -> Path:
    return interim_dir() / "folds.parquet"


def assign_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
    group_col: str = "session_id",
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Return ``df`` with an added integer ``fold`` column (session-grouped).

    Pure function — no IO — so tests can exercise it directly on synthetic data.
    """
    if group_col not in df.columns:
        raise KeyError(f"group column {group_col!r} missing; cannot build safe folds")
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} missing")

    out = df.copy()
    out["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = out[label_col].to_numpy()
    groups = out[group_col].to_numpy()
    for k, (_, val_idx) in enumerate(splitter.split(out, y, groups)):
        out.iloc[val_idx, out.columns.get_loc("fold")] = k

    if (out["fold"] < 0).any():
        raise RuntimeError("fold assignment left unassigned rows — refusing to continue")
    _assert_no_group_overlap(out, group_col)
    return out


def _assert_no_group_overlap(df: pd.DataFrame, group_col: str) -> None:
    """Hard check: no session may appear in more than one fold."""
    per_group_folds = df.groupby(group_col)["fold"].nunique()
    bad = per_group_folds[per_group_folds > 1]
    if len(bad) > 0:
        raise AssertionError(
            f"LEAKAGE: {len(bad)} {group_col}s appear in multiple folds "
            f"(e.g. {list(bad.index[:5])}). Folds must be group-disjoint."
        )


@task(
    "cv.build",
    requires="cpu",
    max_tier="cpu",
    description="generate session-grouped stratified folds once and persist them",
)
def build(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    cfg = get_config()
    out = folds_path()
    if out.is_file() and not force and subsample is None:
        log.info("cv.build: cache hit %s (force=True to redo)", out)
        folds = pd.read_parquet(out)
        return _fold_summary(folds, cached=True, output_path=str(out))

    df = load_train()
    if subsample is not None:
        # subsample by SESSION so the group structure is preserved
        sessions = df["session_id"].drop_duplicates().head(max(1, subsample))
        df = df[df["session_id"].isin(sessions)]

    n_splits = int(cfg.cv["n_splits"])
    folds = assign_folds(
        df,
        n_splits=n_splits,
        seed=cfg.seed,
        group_col=str(cfg.cv["group_col"]),
        label_col=str(cfg.cv["label_col"]),
    )
    keep = ["response_id", "session_id", LABEL_COL, "fold"]
    folds = folds[keep]

    dest = out if subsample is None else interim_dir() / f"folds_sub{subsample}.parquet"
    write_parquet(folds, dest)
    log.info("cv.build: %d rows, %d folds -> %s", len(folds), n_splits, dest)
    return _fold_summary(folds, cached=False, output_path=str(dest))


def _fold_summary(folds: pd.DataFrame, cached: bool, output_path: str) -> dict[str, Any]:
    by = folds.groupby("fold").agg(
        n_rows=("response_id", "size"),
        n_sessions=("session_id", "nunique"),
        pos_rate=(LABEL_COL, "mean"),
    )
    return {
        "output_path": output_path,
        "n_rows": int(len(folds)),
        "n_folds": int(folds["fold"].nunique()),
        "n_sessions": int(folds["session_id"].nunique()),
        "per_fold": by.round(4).to_dict("index"),
        "cached": cached,
    }


def load_folds(subsample: int | None = None) -> pd.DataFrame:
    """Load persisted folds, raising a helpful error if cv.build has not run."""
    path = folds_path() if subsample is None else interim_dir() / f"folds_sub{subsample}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing — run tasks.run('cv.build') first")
    return pd.read_parquet(path)


def fold_indices(folds: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(train_idx, val_idx), ...] positional indices for each fold."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    fold_vals = np.sort(folds["fold"].unique())
    arr = folds["fold"].to_numpy()
    for k in fold_vals:
        val = np.where(arr == k)[0]
        tr = np.where(arr != k)[0]
        out.append((tr, val))
    return out
