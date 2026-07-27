"""Structured logging with rotation.

Logs stream to the console and to a rotating file under ``artifacts/logs/``. The
competition submission is a separate, deliberately quiet path (main.py configures
its own minimal logging and never emits test-data info), so this module is for the
training/experiment side where verbose, timestamped logs are useful.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "traceace") -> logging.Logger:
    """Return a namespaced logger (child of the ``traceace`` root logger)."""
    return logging.getLogger(name if name.startswith("traceace") else f"traceace.{name}")


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the ``traceace`` root logger once (idempotent).

    Console handler + a rotating file handler (5 MB x 3 backups) if ``log_dir`` given.
    """
    global _CONFIGURED
    root = logging.getLogger("traceace")
    # Colab's sync() deliberately removes and re-imports every traceace module before each
    # task. Module globals reset, but Logger objects survive in logging's process-wide
    # registry. Relying only on _CONFIGURED therefore added another console/file handler on
    # every cell and multiplied each line. Store the durable marker on the Logger itself.
    if _CONFIGURED or getattr(root, "_traceace_configured", False):
        _CONFIGURED = True
        return root

    # Upgrade a live notebook that already accumulated handlers under the old code.
    # This namespaced logger is owned entirely by this module.
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level)
    root.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "traceace.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _CONFIGURED = True
    root._traceace_configured = True  # type: ignore[attr-defined]
    return root
