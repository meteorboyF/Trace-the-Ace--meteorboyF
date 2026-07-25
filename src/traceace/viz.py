"""Figure helpers — publication-quality defaults for the write-up.

The write-up (publication bonus) reuses these figures directly, so every figure is
saved as **both PNG and PDF**, with labelled axes readable at print size and a
colourblind-safe palette. Matplotlib uses the non-interactive ``Agg`` backend so
figures render headless on Colab and in CI.

``matplotlib`` is a dev/Colab-only dependency and is never imported by the packaged
submission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Colourblind-safe qualitative palette (Bang Wong, 2011) — safe for print + screen.
PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
]

_SETUP_DONE = False


def setup_mpl() -> None:
    """Apply publication defaults (idempotent)."""
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.autolayout": True,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )
    _SETUP_DONE = True


def save_fig(
    fig: Any, path_stem: Path | str, formats: tuple[str, ...] = ("png", "pdf")
) -> list[Path]:
    """Save ``fig`` as ``<stem>.png`` and ``<stem>.pdf``; return the written paths."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out
