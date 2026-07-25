"""Shared fixtures — synthetic only.

CI runs on GitHub Actions where the competition data does **not** exist (and must never
exist). Every test here builds its own synthetic fixtures, so the suite is fully
runnable without the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import traceace


@pytest.fixture()
def synth_repo(tmp_path, monkeypatch):
    """A minimal repo layout with conf/base.yaml, configured for traceace."""
    repo = tmp_path / "repo"
    (repo / "conf").mkdir(parents=True)
    src = _read_real_base_yaml()
    (repo / "conf" / "base.yaml").write_text(src)
    traceace.configure(repo_dir=repo, drive_root=None, quiet=True)
    return repo


def _read_real_base_yaml() -> str:
    """Use the repo's real base.yaml so tests exercise the actual configuration."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[1] / "conf" / "base.yaml"
    return here.read_text()


@pytest.fixture()
def synth_train() -> pd.DataFrame:
    """Training frame with the real shape: sessions carrying 1..4 responses each."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(120):
        sid = f"sess{s:04d}"
        n_resp = int(rng.integers(1, 5))
        for r in range(n_resp):
            rows.append(
                {
                    "response_id": f"{sid}_r{r}",
                    "session_id": sid,
                    "learning_objective_id": f"lo{int(rng.integers(0, 12)):02d}",
                    "learning_objective": f"objective text {int(rng.integers(0, 12))}",
                    "correct": float(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synth_transcript() -> pd.DataFrame:
    """One transcript with all three roles, relative timestamps, and ASR artifacts."""
    rows = []
    t = 0
    for i in range(60):
        role = ["tutor", "student", "background"][i % 3 if i % 7 == 0 else i % 2]
        content = {
            "tutor": "So what is 2 plus 2? Does that make sense?",
            "student": "Um, I think it's 4. [unclear]",
            "background": "[unclear]",
        }[role]
        t += 3 + (i % 5)
        rows.append(
            {
                "session_id": "sess0000",
                "utterance_id": str(i),
                "role": role,
                "content": content,
                "timestamp": f"{t // 3600}:{(t % 3600) // 60:02d}:{t % 60:02d}",
            }
        )
    return pd.DataFrame(rows)
