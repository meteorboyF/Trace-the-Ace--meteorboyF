"""IO helpers: canonical dataset loaders and parquet read/write.

All readers target the local working disk (never the Drive FUSE mount — that guard
lives in :mod:`traceace.paths`). Parquet is written zstd-compressed; the packaged
submission never touches parquet, so pyarrow is a dev/Colab-only dependency here.

The label column in the real files is ``is_correct``; we canonicalize it to
``correct`` on load so the rest of the codebase speaks one vocabulary
(see docs/DATA.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .paths import raw_file, transcripts_dir

# Canonical internal column names.
LABEL_COL = "correct"
# Accepted source spellings for the label, mapped to LABEL_COL on load.
_LABEL_ALIASES = ("is_correct", "correct", "label")


# --- parquet -----------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: str | Path, compression: str = "zstd") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --- canonical dataset loaders ----------------------------------------------
def load_train_features(config: Config | None = None) -> pd.DataFrame:
    """Load response-level features. Columns include response_id, session_id,
    learning_objective(_id)."""
    df = pd.read_csv(raw_file("train_features", config), dtype=str)
    return df


def subsample_session_ids(n_sessions: int, config: Config | None = None) -> list[str]:
    """The canonical deterministic session cohort for every subsampled task."""
    if n_sessions < 1:
        raise ValueError("n_sessions must be positive")
    feats = load_train_features(config)
    return (
        feats["session_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(n_sessions)
        .tolist()
    )


def load_train_labels(config: Config | None = None) -> pd.DataFrame:
    """Load labels, canonicalizing the label column to ``correct`` (float)."""
    df = pd.read_csv(raw_file("train_labels", config))
    df = _canonicalize_label(df)
    df["response_id"] = df["response_id"].astype(str)
    return df


def load_train(config: Config | None = None) -> pd.DataFrame:
    """Features joined with labels on response_id (inner join)."""
    feats = load_train_features(config)
    labels = load_train_labels(config)
    merged = feats.merge(labels[["response_id", LABEL_COL]], on="response_id", how="inner")
    return merged


def load_submission_format(smoke: bool = False, config: Config | None = None) -> pd.DataFrame:
    """Load the submission format (full test set, or the 100-row smoke set)."""
    key = "submission_format_smoke" if smoke else "submission_format"
    df = pd.read_csv(raw_file(key, config), dtype={"response_id": str})
    return df


def read_transcript(session_id: str, config: Config | None = None) -> pd.DataFrame:
    """Read one session transcript CSV (from the locally-extracted directory)."""
    path = transcripts_dir(config) / f"{session_id}.csv"
    df = pd.read_csv(path, dtype=str)
    return df


def transcript_path(session_id: str, config: Config | None = None) -> Path:
    return transcripts_dir(config) / f"{session_id}.csv"


def _canonicalize_label(df: pd.DataFrame) -> pd.DataFrame:
    for alias in _LABEL_ALIASES:
        if alias in df.columns:
            if alias != LABEL_COL:
                df = df.rename(columns={alias: LABEL_COL})
            df[LABEL_COL] = df[LABEL_COL].astype(float)
            return df
    raise KeyError(
        f"no label column found among {_LABEL_ALIASES!r}; columns were {list(df.columns)}"
    )
