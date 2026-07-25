"""traceace — Trace the Ace competition package.

Public surface used by the Colab notebook and local dev:

    import traceace
    traceace.configure(repo_dir=".", drive_root=None)   # local
    traceace.tasks.run("eda.overview")

All logic lives here; the notebook is a thin, stable wrapper (docs/BRIEF.md §6).
"""

from __future__ import annotations

import os
import time

from . import budget, tasks
from .config import Config, get_config
from .config import configure as _configure_impl
from .runtime import detect_accelerator

__all__ = ["configure", "get_config", "tasks", "budget", "Config"]

_SESSION_START = time.monotonic()


def configure(
    repo_dir: str | os.PathLike[str] = ".",
    drive_root: str | os.PathLike[str] | None = None,
    work_dir: str | os.PathLike[str] | None = None,
    experiment: str | None = None,
    quiet: bool = False,
) -> Config:
    """Configure the process and print the status header (accelerator, budget, staging).

    Mirrors the header the notebook's ``sync()`` expects: accelerator, tier, units/hr,
    session elapsed, cumulative units, and staging status.
    """
    cfg = _configure_impl(
        repo_dir=repo_dir, drive_root=drive_root, work_dir=work_dir, experiment=experiment
    )
    if not quiet:
        _print_status_header(cfg)
    return cfg


def _print_status_header(cfg: Config) -> None:
    acc = detect_accelerator()
    rate = cfg.unit_rates.get(acc.tier, 0.0)
    try:
        spent = budget.cumulative_units()
    except Exception:
        spent = 0.0
    remaining = cfg.total_units - spent
    elapsed_min = (time.monotonic() - _SESSION_START) / 60.0

    staged = _staging_status(cfg)
    mode = "COLAB" if cfg.drive_root is not None else "LOCAL"

    print("┌" + "─" * 62 + "┐")
    print(f"│ traceace configured · mode={mode}")
    print(f"│ accel   : {acc.name} · tier={acc.tier} · ~{rate:g} units/hr")
    print(f"│ budget  : spent {spent:.2f} / {cfg.total_units:.0f} · remaining {remaining:.2f}")
    print(f"│ session : {elapsed_min:.1f} min elapsed")
    print(f"│ staging : {staged}")
    print(f"│ workdir : {cfg.work_dir}")
    print("└" + "─" * 62 + "┘", flush=True)


def _staging_status(cfg: Config) -> str:
    try:
        from .staging import is_staged

        return "staged (transcripts extracted locally)" if is_staged() else "NOT staged"
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"
