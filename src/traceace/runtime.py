"""Accelerator detection and compute tiers.

The whole compute-budget discipline (docs/BRIEF.md §4) hinges on knowing which
runtime we are attached to. ``units burn while the runtime is CONNECTED, not while
it computes`` — so an idle A100 is the single biggest waste vector, and the task
tier-guard exists to refuse CPU work on a paid GPU.

:func:`current_tier` returns one of ``cpu | t4 | l4 | a100 | h100 | other``. Detection
is deliberately dependency-light: we shell out to ``nvidia-smi`` (present on every
Colab GPU runtime) and match the GPU name, falling back to ``torch`` if importable.
Everything degrades gracefully to ``cpu`` when no accelerator is found.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

Tier = str  # one of: cpu, t4, l4, a100, h100, other

_TIER_ORDER: dict[Tier, int] = {
    "cpu": 0,
    "t4": 1,
    "l4": 2,
    "a100": 3,
    "h100": 4,
    "other": 3,  # unknown GPU: treat as "roughly A100-class" for guard purposes
}


@dataclass(frozen=True)
class Accelerator:
    tier: Tier
    name: str  # human-readable GPU name, or "CPU"
    vram_mb: int | None


def _query_nvidia_smi() -> tuple[str, int | None] | None:
    """Return (gpu_name, vram_mb) via nvidia-smi, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else "unknown-gpu"
    vram: int | None = None
    if len(parts) > 1 and parts[1].isdigit():
        vram = int(parts[1])
    return name, vram


def _tier_from_name(name: str) -> Tier:
    n = name.lower()
    if "h100" in n:
        return "h100"
    if "a100" in n:
        return "a100"
    if "l4" in n:
        return "l4"
    if "t4" in n:
        return "t4"
    # V100, A10, etc. — a paid GPU we don't have a rate row for.
    return "other"


def detect_accelerator() -> Accelerator:
    """Detect the attached accelerator. Never raises; falls back to CPU."""
    q = _query_nvidia_smi()
    if q is not None:
        name, vram = q
        return Accelerator(tier=_tier_from_name(name), name=name, vram_mb=vram)

    # Fallback: torch may see a GPU even if nvidia-smi is missing (rare).
    try:
        import torch  # noqa: PLC0415  (optional, GPU-only)

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            return Accelerator(
                tier=_tier_from_name(name),
                name=name,
                vram_mb=int(props.total_memory // (1024 * 1024)),
            )
    except Exception:
        pass

    return Accelerator(tier="cpu", name="CPU", vram_mb=None)


def current_tier() -> Tier:
    return detect_accelerator().tier


def tier_rank(tier: Tier) -> int:
    """Ordinal for comparing tiers. Higher = more capable/expensive."""
    return _TIER_ORDER.get(tier, _TIER_ORDER["other"])


def tier_at_least(have: Tier, need: Tier) -> bool:
    return tier_rank(have) >= tier_rank(need)


def tier_at_most(have: Tier, cap: Tier) -> bool:
    return tier_rank(have) <= tier_rank(cap)
