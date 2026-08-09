"""Shared sparse-text document construction copied into the submission archive."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sparse_text_document(
    df: pd.DataFrame,
    relevant_spans: list[tuple[int, int]],
    max_chars: int = 8000,
    context_utterances: int = 96,
) -> str:
    """Build compact role-aware dialogue without using objective text as a predictor."""
    if df is None or df.empty or max_chars <= 0:
        return ""
    n = len(df)
    if n <= context_utterances:
        context_idx = np.arange(n, dtype=int)
    else:
        context_idx = np.unique(np.linspace(0, n - 1, context_utterances).round().astype(int))
    relevant_idx: set[int] = set()
    for start, stop in relevant_spans:
        relevant_idx.update(range(max(0, int(start)), min(n, int(stop))))

    def render(indices: list[int], marker: str) -> str:
        lines = [marker]
        for index in indices:
            row = df.iloc[index]
            role = str(row.get("role", "unknown")).lower()
            content = str(row.get("content", ""))
            lines.append(f"<{role}> {content}")
        return "\n".join(lines)

    local = render(sorted(relevant_idx), "<relevant_dialogue>") if relevant_idx else ""
    context = render(context_idx.tolist(), "<lesson_context>")
    if len(local) >= max_chars:
        return local[:max_chars]
    return (local + "\n" + context[: max_chars - len(local) - 1]).strip()
