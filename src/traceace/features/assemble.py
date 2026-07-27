"""Assemble feature blocks into the response-level design matrix.

Session-level blocks (structural, linguistic, temporal, embeddings) are joined onto
response rows by ``session_id``; the LO-alignment block is already response-level and
joins by ``response_id``.

**Ablation support is first-class.** ``build_matrix(blocks=[...])`` selects which blocks
to include, so ``interpret.ablation`` can measure each block's marginal contribution and
we can make claims about *what* mattered, not merely *that* something worked (§11).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..io import LABEL_COL
from ..logging_utils import get_logger
from .common import block_cache_path

log = get_logger("features.assemble")

# block name -> (cache stem, version, join key, module that computes it)
# The module is needed to resolve the source-hash component of the cache path, so a loader
# and a builder can never disagree about which cache file "the structural block" means.
BLOCKS: dict[str, tuple[str, str, str, str]] = {
    "structural": ("structural", "v1", "session_id", "structural"),
    "linguistic": ("linguistic", "v1", "session_id", "linguistic"),
    "temporal": ("temporal", "v1", "session_id", "temporal"),
    "lo_alignment": ("lo_alignment", "v1_lexical", "response_id", "lo_alignment"),
    "feedback": ("feedback", "v1", "response_id", "feedback"),
    "trajectory": ("trajectory", "v1", "response_id", "trajectory"),
    # requires features.window_embeddings (GPU, once)
    "content": ("content", "v1_k48_top3", "response_id", "content"),
}


def block_source_hash(name: str) -> str:
    """Resolve the source-hash cache component for a block, via its owning module."""
    from importlib import import_module

    mod = import_module(f".{BLOCKS[name][3]}", package=__package__)
    return str(mod._source())


# Block membership is decided ONLY by `interpret.ablation_repeated` (paired leave-one-out
# across 5 fold assignments, mean ± SD). Single-seed readings below ~1e-3 are noise at this
# sample size and must not drive decisions — we made that mistake once with `temporal`.
#
# Measured paired deltas (positive = block contributes), 5 seeds:
#   trajectory    +0.00226 ± 0.00011   5/5  <- strongest
#   linguistic    +0.00174 ± 0.00025   5/5
#   lo_alignment  -0.00030 ± 0.00024   5/5  <- REMOVED: significantly HURTS
#   feedback      -0.00007 ± 0.00014   2/5  indistinguishable from zero
#   structural    +0.00006 ± 0.00030   4/5  indistinguishable from zero
#   temporal      +0.00005 ± 0.00047   4/5  indistinguishable from zero
#
# `lo_alignment` FEATURES are dropped (the only removal the evidence supports), but the
# MODULE stays: it defines the sliding windows that trajectory/feedback/content are scoped
# to, and its key-moment positions are a reported research finding. It may earn its place
# back once the semantic backend replaces lexical matching.
# Blocks that merely look redundant are deprioritized, NOT deleted — substitutability means
# a block can become useful again when a neighbouring block improves.
DEFAULT_BLOCKS = [
    "structural",
    "linguistic",
    "temporal",
    "feedback",
    "trajectory",
]
# ALL_BLOCKS is what the ablation sweeps, so negative results stay visible in the report.
ALL_BLOCKS = [*DEFAULT_BLOCKS[:3], "lo_alignment", *DEFAULT_BLOCKS[3:]]

# ---------------------------------------------------------------------------
# CROSS-ROW FEATURES — competition-rules exposure. See conf/base.yaml.
# ---------------------------------------------------------------------------
# These are the ONLY features whose value depends on other rows of the feature file
# (they are computed by grouping rows that share a session_id). Everything else is a
# function of (this row's learning objective, this session's transcript, training-fitted
# parameters) and is therefore independent by construction.
#
# Audited exhaustively 2026-07-26 by tracing every feature function's inputs: the sole
# cross-row argument anywhere in the codebase is `all_lo_spans` in
# `inference_lib.lo_position_features`, which feeds exactly these four.
CROSS_ROW_FEATURES: frozenset[str] = frozenset(
    {
        "lopos_n_competing_los",  # count of objectives assessed in the same session
        "lopos_ordinal",  # rank of this objective among the session's objectives
        "lopos_ordinal_frac",  # same, normalized
        "lopos_overlap_with_others",  # overlap of this objective's window with the others'
    }
)


def cross_row_allowed() -> bool:
    """Whether cross-row aggregates may be used as feature inputs (default: NO)."""
    from ..config import get_config

    return bool(get_config().get("features", "allow_cross_row_aggregates", default=False))


# Columns that are identifiers/labels, never features.
NON_FEATURE = {
    "response_id",
    "session_id",
    "fold",
    LABEL_COL,
    "learning_objective",
    "learning_objective_id",
}


def load_block(name: str, subsample: int | None = None) -> pd.DataFrame:
    if name not in BLOCKS:
        raise KeyError(f"unknown feature block {name!r}; known: {sorted(BLOCKS)}")
    stem, version, _, _ = BLOCKS[name]
    path = block_cache_path(stem, version, subsample, source_hash=block_source_hash(name))
    if not path.is_file():
        raise FileNotFoundError(
            f"feature block {name!r} not built ({path}). Run tasks.run('features.{name}')."
        )
    return pd.read_parquet(path)


def build_matrix(
    base: pd.DataFrame,
    blocks: list[str] | None = None,
    subsample: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Join the requested blocks onto ``base`` (which must carry response_id+session_id).

    Returns ``(frame, feature_columns)``.
    """
    blocks = list(DEFAULT_BLOCKS if blocks is None else blocks)
    out = base.copy()
    for name in blocks:
        df = load_block(name, subsample=subsample)
        _, _, key, _ = BLOCKS[name]
        if key not in df.columns:
            raise KeyError(f"feature block {name!r} is missing join key {key!r}")
        if df[key].isna().any():
            raise RuntimeError(f"feature block {name!r} has null {key!r} values")
        if df[key].duplicated().any():
            raise RuntimeError(
                f"feature block {name!r} has duplicate {key!r} values; "
                "joining it would duplicate training rows"
            )
        drop = [c for c in df.columns if c in out.columns and c != key]
        before = len(out)
        new_cols = [c for c in df.columns if c != key and c not in drop]
        out = out.merge(df.drop(columns=drop), on=key, how="left", validate="many_to_one")
        if len(out) != before:
            raise RuntimeError(f"feature block {name!r} changed row count {before} -> {len(out)}")
        if new_cols:
            missing = out[new_cols].isna().all(axis=1)
            if missing.any():
                raise RuntimeError(
                    f"feature block {name!r} is absent for {int(missing.sum())}/{len(out)} "
                    "rows; caches and CV likely used different session cohorts"
                )
        log.info("assemble: joined block %s on %s (%d cols)", name, key, df.shape[1] - 1)

    feat_cols = [c for c in out.columns if c not in NON_FEATURE]
    # Drop rules-risky cross-row aggregates unless explicitly enabled.
    if not cross_row_allowed():
        dropped_cross = [c for c in feat_cols if c in CROSS_ROW_FEATURES]
        if dropped_cross:
            log.info(
                "assemble: EXCLUDING %d cross-row feature(s) for rules safety: %s "
                "(set features.allow_cross_row_aggregates=true to include)",
                len(dropped_cross),
                sorted(dropped_cross),
            )
        feat_cols = [c for c in feat_cols if c not in CROSS_ROW_FEATURES]
    # keep only numeric feature columns
    numeric = out[feat_cols].select_dtypes(include="number").columns.tolist()
    dropped = set(feat_cols) - set(numeric)
    if dropped:
        log.debug("assemble: dropping %d non-numeric columns", len(dropped))
    return out, numeric


def block_of(column: str) -> str:
    """Map a feature column back to its block, via prefix. Used by ablation/importance."""
    for block, prefix in (
        ("structural", "struct_"),
        ("linguistic", "ling_"),
        ("temporal", "temp_"),
        ("lo_alignment", "lo_"),
        ("feedback", "fb_"),
        ("feedback", "fbs_"),
        ("trajectory", "traj_"),
        ("lo_position", "lopos_"),
        ("content", "cont_"),
        ("embeddings", "emb_"),
    ):
        if column.startswith(prefix):
            return block
    return "other"


def summarize(frame: pd.DataFrame, feat_cols: list[str]) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(block_of(c) for c in feat_cols)
    return {
        "n_rows": int(len(frame)),
        "n_features": len(feat_cols),
        "features_by_block": dict(counts),
        "missing_rate": float(frame[feat_cols].isna().to_numpy().mean()) if feat_cols else 0.0,
    }
