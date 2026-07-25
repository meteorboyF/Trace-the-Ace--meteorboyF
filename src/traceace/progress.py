"""Progress instrumentation — tqdm wrappers, heartbeat, and the submission kill-switch.

Two hard requirements from the brief (§5):

1. **Every long loop shows progress with an ETA** so the operator can decide whether
   to let a paid GPU keep running. We wrap ``tqdm.auto`` so it renders in both Colab
   and a terminal, and add a :func:`heartbeat` context manager for long *non-iterative*
   operations (LightGBM fit, model load, zip extraction).
2. **Progress must be HARD-DISABLED in submission mode.** The competition container caps
   logging at 500 lines x 500 chars; tqdm's carriage-return spam can blow through it and
   is disqualification-adjacent. Every bar is gated behind :data:`PROGRESS_ENABLED`, which
   ``main.py`` sets to ``False`` and ``submission.verify`` checks.

The switch is a module-level flag plus an env override (``TRACEACE_PROGRESS=0``) so it can
be forced off from outside the process too.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from tqdm.auto import tqdm

# Master switch. main.py sets this False. Env var forces it off regardless.
PROGRESS_ENABLED: bool = os.environ.get("TRACEACE_PROGRESS", "1") not in {"0", "false", "False"}


def set_progress(enabled: bool) -> None:
    """Enable/disable all progress output process-wide."""
    global PROGRESS_ENABLED
    PROGRESS_ENABLED = bool(enabled)


def progress_enabled() -> bool:
    # Env var wins if explicitly set to a falsey value (belt-and-braces for main.py).
    if os.environ.get("TRACEACE_PROGRESS", "1") in {"0", "false", "False"}:
        return False
    return PROGRESS_ENABLED


def pbar[T](
    iterable: Iterable[T] | None = None,
    *,
    desc: str,
    total: int | None = None,
    leave: bool = True,
    unit: str = "it",
    position: int | None = None,
) -> Any:
    """A tqdm progress bar that is a no-op when progress is disabled.

    Always pass a descriptive ``desc``. Use ``leave=True`` (default) for top-level
    bars so completed work stays visible; nested/inner bars should pass ``leave=False``.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        unit=unit,
        position=position,
        disable=not progress_enabled(),
        dynamic_ncols=True,
        file=sys.stdout,
    )


@contextmanager
def heartbeat(
    label: str,
    every_seconds: float = 30.0,
    enabled: bool | None = None,
) -> Iterator[None]:
    """Print ``label ... <elapsed>s`` every ``every_seconds`` for a long blocking call.

    Use for operations with no natural iteration to hang a bar on: LightGBM ``fit``,
    a model load, a zip extraction. Emits nothing when progress is disabled, so it is
    safe to leave in submission code paths.

    Example
    -------
    >>> with heartbeat("training LightGBM"):
    ...     model.fit(X, y)
    """
    on = progress_enabled() if enabled is None else enabled
    if not on:
        yield
        return

    stop = threading.Event()
    start = time.monotonic()

    def _tick() -> None:
        while not stop.wait(every_seconds):
            elapsed = time.monotonic() - start
            print(f"  … {label}: {elapsed:6.1f}s elapsed", flush=True)

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        if on:
            print(f"  ✓ {label}: {time.monotonic() - start:.1f}s", flush=True)
