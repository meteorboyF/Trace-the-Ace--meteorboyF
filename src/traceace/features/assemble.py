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

# block name -> (cache stem, version, join key)
BLOCKS: dict[str, tuple[str, str, str]] = {
    "structural": ("structural", "v1", "session_id"),
    "linguistic": ("linguistic", "v1", "session_id"),
    "temporal": ("temporal", "v1", "session_id"),
    "lo_alignment": ("lo_alignment", "v1_lexical", "response_id"),
    "feedback": ("feedback", "v1", "response_id"),
    "trajectory": ("trajectory", "v1", "response_id"),
    # requires features.window_embeddings (GPU, once)
    "content": ("content", "v1_k48", "response_id"),
}

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
    stem, version, _ = BLOCKS[name]
    path = block_cache_path(stem, version, subsample)
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
        _, _, key = BLOCKS[name]
        drop = [c for c in df.columns if c in out.columns and c != key]
        out = out.merge(df.drop(columns=drop), on=key, how="left")
        log.info("assemble: joined block %s on %s (%d cols)", name, key, df.shape[1] - 1)

    feat_cols = [c for c in out.columns if c not in NON_FEATURE]
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
