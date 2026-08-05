from __future__ import annotations

import pandas as pd

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
        transformers.AutoModel, "from_pretrained", lambda _: transformers.BertModel(config)
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
