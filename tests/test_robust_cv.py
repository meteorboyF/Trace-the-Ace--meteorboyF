from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traceace.io import LABEL_COL
from traceace.robust_cv import assign_objective_folds, purged_split_indices


def _frame() -> pd.DataFrame:
    rows = []
    for objective in range(10):
        for sample in range(4):
            # s0 is deliberately shared by two objectives: purging must remove it.
            session = "s0" if objective in {0, 1} and sample == 0 else f"s{objective}_{sample}"
            rows.append(
                {
                    "response_id": f"r{objective}_{sample}",
                    "session_id": session,
                    "learning_objective_id": f"lo{objective}",
                    LABEL_COL: float((objective + sample) % 2),
                }
            )
    return pd.DataFrame(rows)


def test_objective_folds_keep_each_objective_whole():
    folds = assign_objective_folds(_frame(), n_splits=5, seed=7)
    assert folds["fold"].nunique() == 5
    assert folds.groupby("learning_objective_id")["fold"].nunique().max() == 1


def test_purged_objective_split_has_no_session_or_objective_overlap():
    folds = assign_objective_folds(_frame(), n_splits=5, seed=7)
    for fold in sorted(folds["fold"].unique()):
        train_idx, valid_idx = purged_split_indices(folds, int(fold))
        train, valid = folds.iloc[train_idx], folds.iloc[valid_idx]
        assert set(train["session_id"]).isdisjoint(valid["session_id"])
        assert set(train["learning_objective_id"]).isdisjoint(valid["learning_objective_id"])


def test_gbdt_objective_masks_apply_the_same_session_and_objective_purge():
    from traceace.models.gbdt import _outer_masks

    folds = assign_objective_folds(_frame(), n_splits=5, seed=7)
    for fold in sorted(folds["fold"].unique()):
        train, valid = _outer_masks(folds, "objective", int(fold))
        assert set(folds.loc[train, "session_id"]).isdisjoint(folds.loc[valid, "session_id"])
        assert set(folds.loc[train, "learning_objective_id"]).isdisjoint(
            folds.loc[valid, "learning_objective_id"]
        )


def test_full_objective_fold_health_rejects_degenerate_assignment():
    from traceace.robust_cv import _validate_folds

    rows = []
    for objective in range(100):
        # One fold gets only one objective; the other four receive the remainder.
        fold = 0 if objective == 0 else 1 + ((objective - 1) % 4)
        rows.append(
            {
                "response_id": f"r{objective}",
                "session_id": f"s{objective}",
                "learning_objective_id": f"lo{objective}",
                LABEL_COL: float(objective % 2),
                "fold": fold,
            }
        )
    with pytest.raises(RuntimeError, match="degenerate"):
        _validate_folds(pd.DataFrame(rows), "objective")


def test_transformer_robust_split_cannot_expand_a_smoke_cohort(monkeypatch):
    from traceace.models.hierarchical_transformer import _split_frames

    full = assign_objective_folds(_frame(), n_splits=5, seed=7)
    smoke_ids = set(full["response_id"].head(12))
    smoke = _frame()[lambda frame: frame["response_id"].isin(smoke_ids)]
    monkeypatch.setattr(
        "traceace.models.hierarchical_transformer.load_robust_folds", lambda kind: full
    )
    train, valid = _split_frames(smoke, "objective", int(full.iloc[0]["fold"]))
    assert set(train["response_id"]) | set(valid["response_id"]) <= smoke_ids
    assert not train[LABEL_COL].isna().any()
    assert not valid[LABEL_COL].isna().any()


def test_hierarchical_transformer_dual_heads_mask_padding(monkeypatch):
    torch = __import__("pytest").importorskip("torch")
    transformers = __import__("pytest").importorskip("transformers")
    from traceace.models.hierarchical_transformer import HierarchicalEncoder

    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda _, **kwargs: transformers.BertModel(config),
    )
    model = HierarchicalEncoder.build("local-test", dropout=0.0, gradient_checkpointing=False)
    chunks = torch.randint(0, 32, (2, 3, 8))
    masks = torch.ones_like(chunks)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    objectives = torch.randint(0, 32, (2, 5))
    objective_mask = torch.ones_like(objectives)
    model.eval()
    with torch.no_grad():
        plain, conditional = model(chunks, masks, valid, objectives, objective_mask)
    assert plain.shape == conditional.shape == (2,)
    assert torch.isfinite(plain).all() and torch.isfinite(conditional).all()


def test_transformer_collator_covers_whole_lesson_with_compact_token_cache():
    torch = __import__("pytest").importorskip("torch")
    from traceace.models.hierarchical_transformer import _Collator

    class Tokenizer:
        pad_token_id = 0
        cls_token_id = 98
        sep_token_id = 99
        bos_token_id = None
        eos_token_id = None

        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(range(40))

        @staticmethod
        def __call__(texts, **kwargs):
            return {
                "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
            }

    collator = _Collator(Tokenizer(), chunk_tokens=10, max_chunks=3, objective_tokens=8)
    ids, _, valid, *_ = collator(
        [{"text": "lesson", "objective": "obj", "label": 1.0, "response_id": "r1"}]
    )
    assert ids[0, :, 1].tolist() == [0, 16, 32]
    assert valid.tolist() == [[True, True, True]]
    assert collator.cache["lesson"].dtype == np.int32


def test_checkpoint_state_preserves_integer_buffers_and_halves_floats():
    torch = __import__("pytest").importorskip("torch")
    from traceace.models.hierarchical_transformer import _checkpoint_state

    module = torch.nn.Linear(3, 2)
    module.register_buffer("indices", torch.tensor([0, 4097], dtype=torch.long))
    state = _checkpoint_state(module)
    assert state["weight"].dtype == torch.float16
    assert state["indices"].dtype == torch.long
    assert state["indices"].tolist() == [0, 4097]
