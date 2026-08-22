"""Staging between Google Drive and the local working disk.

The storage rule (docs/BRIEF.md §8): **Drive holds a handful of large files; the
local SSD (`/content/work`) is the working disk.** Reading thousands of small
transcript CSVs directly off the Drive FUSE mount would take hours. So:

* :func:`stage_local` copies the single raw archive Drive -> local, extracts it
  **locally**, and is a no-op if already staged. Re-staging after a Colab reset
  must cost under a minute (one big sequential copy, not per-file).
* :func:`sync_to_drive` is the ONLY function that writes to Drive, and it tars
  first so we push a handful of large files, never a directory tree.
* Extraction *to* Drive is impossible by construction: extraction always targets
  the local working root, and :func:`~traceace.paths.assert_not_drive` guards the
  iteration paths.

Locally (``drive_root is None``) staging is a no-op: the data already sits in
``data/raw/`` on the working disk.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

from .config import get_config
from .logging_utils import get_logger
from .paths import assert_not_drive, raw_dir, transcripts_dir, work_root
from .progress import heartbeat, pbar

log = get_logger("staging")


def _drive_raw_zip() -> Path | None:
    """Path to ``data/raw.zip`` on Drive, or None when running locally."""
    cfg = get_config()
    if cfg.drive_root is None:
        return None
    return cfg.drive_root / "data" / "raw.zip"


def is_staged() -> bool:
    """True if the transcripts directory is already extracted locally."""
    tdir = transcripts_dir()
    if not tdir.is_dir():
        return False
    # cheap check: at least one csv present
    try:
        next(tdir.glob("*.csv"))
        return True
    except StopIteration:
        return False


def stage_local(force: bool = False) -> Path:
    """Ensure raw data is present and extracted on the local working disk.

    Returns the local ``data/raw`` directory. No-op if already staged (loud skip).

    On Colab: copies ``<drive>/data/raw.zip`` to the local working root and extracts
    it there. Locally: verifies ``data/raw/`` already holds the files (and extracts
    the transcripts zip if the directory is missing).
    """
    cfg = get_config()
    rdir = raw_dir()
    rdir.mkdir(parents=True, exist_ok=True)

    if is_staged() and not force:
        log.info("stage_local: already staged at %s (skip; force=True to redo)", rdir)
        return rdir

    drive_zip = _drive_raw_zip()
    if drive_zip is not None and drive_zip.is_file():
        # Colab path: one big sequential copy off Drive, then extract locally.
        local_zip = work_root() / "raw.zip"
        log.info("stage_local: copying %s -> %s", drive_zip, local_zip)
        with heartbeat("copy raw.zip from Drive"):
            shutil.copyfile(drive_zip, local_zip)
        _extract_archive(local_zip, rdir)
        _flatten_outer_archive(rdir, cfg.canonical_files.values())

    # raw.zip contains the original transcript archive as one of its five members.
    # Extracting the outer archive is therefore only half of staging. This used to run
    # only in local mode, so a fresh Colab stopped with "no transcript CSVs" immediately
    # after successfully copying and extracting raw.zip.
    tzip = rdir / cfg.canonical_files["transcripts_zip"]
    if tzip.is_file() and not is_staged():
        _extract_archive(tzip, transcripts_dir().parent, into_named_dir=True)

    if not is_staged():
        raise FileNotFoundError(
            f"staging completed but no transcript CSVs were found under {transcripts_dir()}"
        )

    log.info("stage_local: staged at %s", rdir)
    return rdir


def _flatten_outer_archive(rdir: Path, canonical_names: Any) -> None:
    """Move canonical members out of one Drive-created wrapper directory.

    Depending on how Drive creates an archive, the same five files may be stored at the ZIP
    root or below a folder such as ``raw/``. Downstream paths are intentionally canonical,
    so normalize either shape immediately after extraction.
    """
    for name in canonical_names:
        destination = rdir / str(name)
        if destination.exists():
            continue
        matches = [path for path in rdir.rglob(str(name)) if path.is_file()]
        if len(matches) > 1:
            raise RuntimeError(f"raw.zip contains multiple candidates for {name}")
        if matches:
            shutil.move(str(matches[0]), destination)


def _extract_archive(archive: Path, dest_dir: Path, into_named_dir: bool = False) -> None:
    """Extract a zip to a LOCAL destination. Refuses to extract onto Drive."""
    dest_dir = assert_not_drive(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        # If every member is a bare "<session>.csv", drop them into a named subdir
        # so the transcripts live under data/raw/train_transcripts/.
        target = dest_dir
        if into_named_dir:
            target = transcripts_dir()
            target.mkdir(parents=True, exist_ok=True)
        for m in pbar(members, desc=f"extract {archive.name}", unit="file"):
            zf.extract(m, target)


def _assert_drive_alive(drive_root: Path) -> None:
    """Refuse to 'sync' into a Drive mount that is no longer real.

    **The failure this exists for, observed 2026-08-22:** an overnight A100 session's Drive
    mount died mid-run. ``/content/drive`` degraded into a plain local directory, so every
    ``copyfile`` into it SUCCEEDED and logged "wrote ..." — and all of it evaporated with
    the runtime. Four trained encoder folds (~58 units) were lost while the logs showed
    five successful syncs. A sync that cannot fail is worse than no sync: it converts a
    recoverable interruption into silent total loss.

    Two cheap checks, both against that exact failure mode:

    * the Colab mountpoint must still BE a mountpoint (a dead mount is a plain dir), and
    * a sentinel write-and-readback through the mount must round-trip.

    Neither proves the bytes reached Google's servers (FUSE uploads are asynchronous; only
    ``drive.flush_and_unmount()`` at shutdown forces that), but they catch the
    phantom-directory case, which is the one that actually burned us.
    """
    mount_root = Path("/content/drive")
    if mount_root != drive_root and mount_root not in drive_root.parents:
        return  # not a Colab-style Drive path (e.g. tests with a tmp dir)
    if not os.path.ismount(mount_root):
        raise RuntimeError(
            f"{mount_root} is NOT a live mount — Google Drive has disconnected. Writes "
            "would land in a phantom local directory and vanish with the runtime. "
            "Re-run drive.mount() (the notebook setup cell) before syncing."
        )
    probe = drive_root / ".sync_liveness"
    token = str(time.time_ns())
    try:
        probe.write_text(token)
        echoed = probe.read_text()
    except OSError as exc:
        raise RuntimeError(f"Drive liveness probe failed ({exc}) — refusing to sync") from exc
    if echoed != token:
        raise RuntimeError("Drive liveness probe did not round-trip — refusing to sync")


def sync_to_drive(local_path: Path, drive_rel: str, as_tar: bool = True) -> Path | None:
    """Push a local file/dir to Drive as a SINGLE archive (the only Drive writer).

    Parameters
    ----------
    local_path:
        Local file or directory to sync.
    drive_rel:
        Destination path relative to ``drive_root`` (e.g. ``artifacts/rollup.tar.zst``).
    as_tar:
        If ``local_path`` is a directory, tar it first so we write one big file to
        Drive instead of walking a tree over FUSE.

    Returns the Drive destination path, or ``None`` when running locally.
    """
    cfg = get_config()
    if cfg.drive_root is None:
        log.info("sync_to_drive: local mode, nothing to sync (%s)", local_path)
        return None
    _assert_drive_alive(cfg.drive_root)

    dest = cfg.drive_root / drive_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    local_path = Path(local_path)

    if local_path.is_dir() and as_tar:
        tmp_tar = work_root() / (local_path.name + ".tar")
        log.info("sync_to_drive: tarring %s -> %s", local_path, tmp_tar)
        with heartbeat("tar for Drive sync"), tarfile.open(tmp_tar, "w") as tf:
            tf.add(local_path, arcname=local_path.name)
        with heartbeat("write tar to Drive"):
            shutil.copyfile(tmp_tar, dest)
    else:
        with heartbeat("write file to Drive"):
            shutil.copyfile(local_path, dest)
    log.info("sync_to_drive: wrote %s", dest)
    return dest


def restore_from_drive(
    drive_rel: str,
    force: bool = False,
    destination_parent: str | None = None,
) -> Path | None:
    """Restore one sync tarball into the local work root on a fresh Colab runtime."""
    cfg = get_config()
    if cfg.drive_root is None:
        return None
    source = cfg.drive_root / drive_rel
    if not source.is_file():
        return None
    with tarfile.open(source) as tf:
        members = tf.getmembers()
        roots = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
        if len(roots) != 1:
            raise RuntimeError(f"cache archive {source} does not have exactly one root")
        extract_root = cfg.work_dir / destination_parent if destination_parent else cfg.work_dir
        destination = extract_root / next(iter(roots))
        # tasks.run() calls ensure_dirs() before every task, so artifacts/, runs/, and cache
        # directories already exist even on a fresh runtime. Existence is not evidence that
        # their Drive archive was restored; the old guard skipped every archive and silently
        # forced full recomputation. Tar extraction is an intentional merge/overwrite.
        extract_root.mkdir(parents=True, exist_ok=True)
        with heartbeat("restore cache from Drive"):
            tf.extractall(extract_root, filter="data")
    log.info("restore_from_drive: restored %s", destination)
    return destination


class CheckpointSyncer:
    """Sync an output on a time interval so a disconnect costs at most one interval.

    Usage::

        cp = CheckpointSyncer(interval_seconds=600)
        for i, batch in enumerate(batches):
            ...
            cp.maybe_sync(local_dir, "artifacts/embeddings")
    """

    def __init__(self, interval_seconds: float = 600.0) -> None:
        self.interval = interval_seconds
        self._last = time.monotonic()

    def maybe_sync(self, local_path: Path, drive_rel: str) -> bool:
        now = time.monotonic()
        if now - self._last >= self.interval:
            sync_to_drive(local_path, drive_rel)
            self._last = now
            return True
        return False
