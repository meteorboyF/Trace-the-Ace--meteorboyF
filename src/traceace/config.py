"""Global configuration for traceace.

The whole package is driven by a single process-wide :class:`Config` object,
established once per process by :func:`configure` (called from the notebook's
``sync()`` or from ``traceace.configure(...)`` locally). Every other module reads
config through :func:`get_config` so there is exactly one source of truth for
paths, seed, cv settings and the compute-unit rate table.

Design choices worth knowing:

* **Local vs Colab are the same code path.** Locally ``drive_root`` is ``None`` and
  the working root is the repo dir. On Colab ``drive_root`` points at the Drive
  mount and the working root is the local SSD (``/content/work``). Nothing else in
  the codebase needs to branch on environment.
* **Fail fast, never silently default.** A missing ``conf/base.yaml`` or an
  unconfigured process raises loudly rather than guessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Process-wide singleton. Set by configure(); read by get_config().
_CONFIG: Config | None = None


@dataclass
class Config:
    """Resolved configuration for one process.

    Attributes
    ----------
    repo_dir:
        Root of the git checkout (where ``conf/`` and ``src/`` live).
    drive_root:
        Google Drive mount root, or ``None`` when running locally. Used by the
        Drive-guard in :mod:`traceace.paths` to forbid catastrophic per-file
        iteration over the FUSE mount.
    work_dir:
        The fast local working disk. Locally this equals ``repo_dir``; on Colab it
        is a local-SSD path (e.g. ``/content/work``). All data/artifact IO targets
        this directory.
    raw:
        Full dict parsed from ``conf/base.yaml`` (plus any experiment overrides).
    """

    repo_dir: Path
    drive_root: Path | None
    work_dir: Path
    raw: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors (typed, so call sites don't index raw dicts) -----
    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def predict_clip_eps(self) -> float:
        return float(self.raw["predict_clip_eps"])

    @property
    def cv(self) -> dict[str, Any]:
        return dict(self.raw["cv"])

    @property
    def budget(self) -> dict[str, Any]:
        return dict(self.raw["budget"])

    @property
    def unit_rates(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["budget"]["unit_rates"].items()}

    @property
    def total_units(self) -> float:
        return float(self.raw["budget"]["total_units"])

    @property
    def heartbeat_seconds(self) -> float:
        return float(self.raw["progress"]["heartbeat_seconds"])

    @property
    def canonical_files(self) -> dict[str, str]:
        return dict(self.raw["canonical_files"])

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: ``cfg.get("embeddings", "model_name")``."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def _load_base_yaml(repo_dir: Path) -> dict[str, Any]:
    base = repo_dir / "conf" / "base.yaml"
    if not base.is_file():
        raise FileNotFoundError(
            f"conf/base.yaml not found under {repo_dir!s}. traceace refuses to run "
            "without an explicit configuration (fail-fast policy)."
        )
    with base.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{base!s} did not parse to a mapping.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def configure(
    repo_dir: str | os.PathLike[str] = ".",
    drive_root: str | os.PathLike[str] | None = None,
    work_dir: str | os.PathLike[str] | None = None,
    experiment: str | None = None,
) -> Config:
    """Establish the process-wide configuration.

    Parameters
    ----------
    repo_dir:
        Path to the repo checkout. Defaults to the current directory (local dev).
    drive_root:
        Drive mount root on Colab; ``None`` locally.
    work_dir:
        Fast working disk. Defaults to ``repo_dir`` locally, and callers on Colab
        pass ``/content/work``. When ``drive_root`` is set but ``work_dir`` is not,
        we default to ``/content/work`` to keep IO off the FUSE mount.
    experiment:
        Optional name of a ``conf/experiments/<name>.yaml`` to merge over base.
    """
    global _CONFIG
    repo = Path(repo_dir).resolve()
    raw = _load_base_yaml(repo)

    if experiment:
        exp_path = repo / "conf" / "experiments" / f"{experiment}.yaml"
        if not exp_path.is_file():
            raise FileNotFoundError(f"experiment config not found: {exp_path!s}")
        with exp_path.open("r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})

    drive = Path(drive_root).resolve() if drive_root is not None else None

    if work_dir is not None:
        work = Path(work_dir).resolve()
    elif drive is not None:
        # On Colab: keep all IO on local SSD, never the Drive FUSE mount.
        work = Path("/content/work")
    else:
        work = repo

    _CONFIG = Config(repo_dir=repo, drive_root=drive, work_dir=work, raw=raw)
    return _CONFIG


def get_config() -> Config:
    """Return the process configuration, or raise if :func:`configure` was skipped."""
    if _CONFIG is None:
        raise RuntimeError(
            "traceace is not configured. Call traceace.configure(repo_dir=..., "
            "drive_root=...) before running any task."
        )
    return _CONFIG


def is_configured() -> bool:
    return _CONFIG is not None
