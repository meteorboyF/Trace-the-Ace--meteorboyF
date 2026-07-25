"""Compute-unit ledger.

Every task run appends a line to ``runs/budget.jsonl`` recording the tier, wall
time, and estimated units consumed, plus a running cumulative. ``budget.report``
summarizes spend by task and tier against the configured balance (733 units).

Unit rates come from ``conf/base.yaml`` (``budget.unit_rates``) and are **never
hardcoded** — Colab's live rates must be reconciled there. Units are estimated as
``rate_per_hour * wall_hours`` because, crucially, *units burn while the runtime is
connected*, so wall time (not compute time) is the correct basis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import get_config
from .paths import runs_dir
from .runtime import Tier


@dataclass
class BudgetEntry:
    ts: str  # ISO8601 UTC
    task: str
    tier: Tier
    wall_seconds: float
    units: float
    cumulative_units: float
    git_sha: str
    extra: dict[str, Any]


def _ledger_path() -> Path:
    d = runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "budget.jsonl"


def units_for(tier: Tier, wall_seconds: float) -> float:
    """Estimated compute units for ``wall_seconds`` connected at ``tier``."""
    rate = get_config().unit_rates.get(tier, 0.0)
    return rate * (wall_seconds / 3600.0)


def cumulative_units() -> float:
    """Sum of all units recorded so far (0.0 if no ledger yet)."""
    path = _ledger_path()
    if not path.is_file():
        return 0.0
    total = 0.0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line).get("units", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return total


def record(
    task: str,
    tier: Tier,
    wall_seconds: float,
    git_sha: str = "unknown",
    **extra: Any,
) -> BudgetEntry:
    """Append one entry to the ledger and return it."""
    units = units_for(tier, wall_seconds)
    cumulative = cumulative_units() + units
    entry = BudgetEntry(
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        task=task,
        tier=tier,
        wall_seconds=round(wall_seconds, 3),
        units=round(units, 4),
        cumulative_units=round(cumulative, 4),
        git_sha=git_sha,
        extra=extra,
    )
    with _ledger_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry)) + "\n")
    return entry


def report() -> dict[str, Any]:
    """Summarize spend by task and tier against the configured balance."""
    cfg = get_config()
    path = _ledger_path()
    by_task: dict[str, float] = defaultdict(float)
    by_tier: dict[str, float] = defaultdict(float)
    total = 0.0
    n = 0
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = float(rec.get("units", 0.0))
                by_task[rec.get("task", "?")] += u
                by_tier[rec.get("tier", "?")] += u
                total += u
                n += 1
    balance = cfg.total_units - total
    return {
        "total_units_configured": cfg.total_units,
        "units_spent": round(total, 3),
        "units_remaining": round(balance, 3),
        "n_runs": n,
        "by_task": {k: round(v, 3) for k, v in sorted(by_task.items(), key=lambda kv: -kv[1])},
        "by_tier": {k: round(v, 3) for k, v in sorted(by_tier.items(), key=lambda kv: -kv[1])},
    }


def format_report(rep: dict[str, Any] | None = None) -> str:
    """Human-readable budget report string (used by budget.report task)."""
    rep = rep or report()
    lines = [
        "=" * 56,
        "COMPUTE BUDGET",
        "=" * 56,
        f"  configured : {rep['total_units_configured']:.0f} units",
        f"  spent      : {rep['units_spent']:.2f} units  ({rep['n_runs']} runs)",
        f"  remaining  : {rep['units_remaining']:.2f} units",
        "-" * 56,
        "  by tier:",
    ]
    for tier, u in rep["by_tier"].items():
        lines.append(f"     {tier:6} {u:8.2f}")
    lines.append("  by task (top spenders):")
    for task, u in list(rep["by_task"].items())[:12]:
        lines.append(f"     {task:28} {u:8.2f}")
    lines.append("=" * 56)
    return "\n".join(lines)
