"""Path resolution and the Drive-mount guard.

Google Drive on Colab is a FUSE mount: **every file operation costs 100-300ms**
regardless of size. Iterating thousands of small files over it (a transcript
directory, a glob of feature shards) is catastrophic — minutes to hours of pure
syscall latency. The rule (docs/BRIEF.md §8) is: Drive holds a *handful of large
files*; the local SSD is the working disk.

This module encodes that rule so it cannot be violated by accident:

* Canonical path accessors (:func:`raw_dir`, :func:`interim_dir`, ...) always
  resolve **under the local working root**, never Drive.
* :func:`assert_not_drive` raises if a Drive path reaches a file-iterating
  function. :func:`iter_files` routes all directory iteration through that guard,
  so "walk this directory" over a Drive path is impossible by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import Config, get_config


class DrivePathError(RuntimeError):
    """Raised when a Drive (FUSE) path is used for per-file iteration."""


def _cfg(config: Config | None) -> Config:
    return config if config is not None else get_config()


# --- Drive guard -------------------------------------------------------------
def is_drive_path(path: str | Path, config: Config | None = None) -> bool:
    """True if ``path`` lives under the configured Drive root."""
    cfg = _cfg(config)
    if cfg.drive_root is None:
        return False
    try:
        Path(path).resolve().relative_to(cfg.drive_root)
        return True
    except ValueError:
        return False


def assert_not_drive(path: str | Path, config: Config | None = None) -> Path:
    """Return ``path`` as a :class:`Path`, or raise if it is on the Drive mount.

    Call this at the top of any function that will iterate a directory, glob, or
    otherwise touch many files. It is the backstop that keeps us off the FUSE mount.
    """
    p = Path(path)
    if is_drive_path(p, config):
        raise DrivePathError(
            f"Refusing to iterate files on the Drive FUSE mount: {p!s}\n"
            "Drive is ~100-300ms per file op. Stage the archive to the local SSD "
            "(staging.stage_local) and iterate there instead."
        )
    return p


def iter_files(
    directory: str | Path,
    pattern: str = "*",
    config: Config | None = None,
) -> Iterator[Path]:
    """Iterate files in ``directory`` — but never over a Drive path.

    All directory walking in the codebase should go through here so the Drive
    guard is unavoidable.
    """
    d = assert_not_drive(directory, config)
    yield from sorted(d.glob(pattern))


# --- Working-root accessors (always local SSD) -------------------------------
def work_root(config: Config | None = None) -> Path:
    return _cfg(config).work_dir


def _sub(key: str, config: Config | None = None) -> Path:
    cfg = _cfg(config)
    rel = cfg.raw["paths"][key]
    p = cfg.work_dir / rel
    return p


def raw_dir(config: Config | None = None) -> Path:
    return _sub("raw", config)


def interim_dir(config: Config | None = None) -> Path:
    return _sub("interim", config)


def features_dir(config: Config | None = None) -> Path:
    return _sub("features", config)


def models_dir(config: Config | None = None) -> Path:
    return _sub("models", config)


def oof_dir(config: Config | None = None) -> Path:
    return _sub("oof", config)


def figures_dir(config: Config | None = None) -> Path:
    return _sub("figures", config)


def logs_dir(config: Config | None = None) -> Path:
    return _sub("logs", config)


def runs_dir(config: Config | None = None) -> Path:
    return _sub("runs", config)


def submission_dir(config: Config | None = None) -> Path:
    return _sub("submission", config)


def ensure_dirs(config: Config | None = None) -> None:
    """Create all standard output directories under the working root."""
    for fn in (
        interim_dir,
        features_dir,
        models_dir,
        oof_dir,
        figures_dir,
        logs_dir,
        runs_dir,
        submission_dir,
    ):
        fn(config).mkdir(parents=True, exist_ok=True)


# --- Canonical raw-file paths ------------------------------------------------
def raw_file(key: str, config: Config | None = None) -> Path:
    """Path to a canonical raw file by config key (e.g. ``train_features``)."""
    cfg = _cfg(config)
    name = cfg.canonical_files[key]
    return raw_dir(config) / name


def transcripts_dir(config: Config | None = None) -> Path:
    """Directory of per-session transcript CSVs (extracted locally)."""
    cfg = _cfg(config)
    return raw_dir(config) / cfg.canonical_files["transcripts_dir"]
