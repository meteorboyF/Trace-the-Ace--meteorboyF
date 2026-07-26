"""Data ingest and transcript consolidation.

``data.ingest`` normalizes the browser-suffixed download filenames to canonical
names by **content shape**, not exact name (the suffixes are random), and validates
schemas. The two ``submission_format`` files are distinguished by **row count**, not
filename — confusing them would waste one of three weekly submissions (§3).

``data.consolidate`` reads the per-session transcript CSVs (extracted locally) into a
single zstd parquet, once. All downstream feature code reads that one file
sequentially rather than doing 22k small reads over and over.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import is_cached
from .io import subsample_session_ids, write_parquet
from .logging_utils import get_logger
from .paths import interim_dir, iter_files, raw_dir, raw_file, transcripts_dir
from .progress import pbar
from .staging import stage_local
from .tasks import task

log = get_logger("data")


# ---------------------------------------------------------------------------
# data.ingest
# ---------------------------------------------------------------------------
@dataclass
class IngestResult:
    renamed: dict[str, str]
    validated: dict[str, list[str]]
    warnings: list[str]


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline().strip()
    return [c.strip() for c in first.split(",")]


def _count_data_rows(path: Path) -> int:
    with path.open("rb") as fh:
        n = sum(1 for _ in fh)
    return max(0, n - 1)  # minus header


def _classify_csv(path: Path) -> str | None:
    """Return a canonical key for a CSV by inspecting its header/shape."""
    cols = set(_read_header(path))
    if {"response_id", "session_id"} <= cols and any(
        c.startswith("learning_objective") for c in cols
    ):
        return "train_features"
    if "response_id" in cols and ({"is_correct", "correct", "label"} & cols):
        return "train_labels"
    if cols == {"response_id", "probability"} or {"response_id", "probability"} <= cols:
        # full vs smoke decided by row count downstream
        return "submission_format_any"
    return None


@task(
    "data.ingest",
    requires="cpu",
    max_tier="cpu",
    description="normalize suffixed download filenames by content shape; validate schemas",
)
def ingest(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    rdir = raw_dir()
    rdir.mkdir(parents=True, exist_ok=True)

    renamed: dict[str, str] = {}
    warnings: list[str] = []

    # --- classify CSVs ------------------------------------------------------
    fmt_candidates: list[Path] = []
    for path in iter_files(rdir, "*.csv"):
        key = _classify_csv(path)
        if key is None:
            warnings.append(f"unclassified csv: {path.name}")
            continue
        if key == "submission_format_any":
            fmt_candidates.append(path)
            continue
        _rename_to(path, raw_file(key), renamed, force)

    # --- distinguish full vs smoke submission format by ROW COUNT -----------
    if fmt_candidates:
        counted = sorted(((p, _count_data_rows(p)) for p in fmt_candidates), key=lambda x: -x[1])
        # largest -> full; the rest (smallest) -> smoke
        _rename_to(counted[0][0], raw_file("submission_format"), renamed, force)
        if len(counted) > 1:
            _rename_to(counted[-1][0], raw_file("submission_format_smoke"), renamed, force)

    # --- transcripts zip ----------------------------------------------------
    for path in iter_files(rdir, "*.zip"):
        if re.match(r"train_transcripts", path.name):
            _rename_to(path, raw_file("transcripts_zip"), renamed, force)

    # --- validate -----------------------------------------------------------
    validated = _validate_canonical(warnings)

    result = IngestResult(renamed=renamed, validated=validated, warnings=warnings)
    for w in warnings:
        log.warning("ingest: %s", w)
    log.info("ingest: renamed %d files", len(renamed))
    return {
        "renamed": result.renamed,
        "validated": result.validated,
        "warnings": result.warnings,
        "n_renamed": len(renamed),
    }


def _rename_to(src: Path, dest: Path, renamed: dict[str, str], force: bool) -> None:
    if src.resolve() == dest.resolve():
        return
    if dest.exists() and not force:
        log.info("ingest: %s already exists (skip); leaving %s in place", dest.name, src.name)
        return
    # copy (not move) so the original suffixed download is preserved as a safety net;
    # both are gitignored, so this costs only local disk.
    shutil.copyfile(src, dest)
    renamed[src.name] = dest.name
    log.info("ingest: %s -> %s", src.name, dest.name)


def _validate_canonical(warnings: list[str]) -> dict[str, list[str]]:
    """Confirm canonical files exist and carry the expected columns."""
    checks = {
        "train_features": {"response_id", "session_id"},
        "train_labels": {"response_id"},
        "submission_format": {"response_id", "probability"},
    }
    out: dict[str, list[str]] = {}
    for key, required in checks.items():
        path = raw_file(key)
        if not path.is_file():
            warnings.append(f"missing canonical file: {path.name}")
            continue
        cols = _read_header(path)
        out[key] = cols
        missing = required - set(cols)
        if missing:
            warnings.append(f"{path.name} missing columns {missing}")
    return out


# ---------------------------------------------------------------------------
# data.consolidate
# ---------------------------------------------------------------------------
_TS_RE = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?\s*$")


def parse_elapsed_seconds(ts: str) -> float:
    """Parse a relative ``H:MM:SS[.f]`` timestamp to seconds. NaN if unparseable."""
    if not isinstance(ts, str):
        return float("nan")
    m = _TS_RE.match(ts)
    if not m:
        return float("nan")
    h, mm, ss, frac = m.groups()
    total = float(int(h) * 3600 + int(mm) * 60 + int(ss))
    if frac:
        total += float("0." + frac)
    return total


def consolidated_path() -> Path:
    return interim_dir() / "transcripts.parquet"


@task(
    "data.consolidate",
    requires="cpu",
    max_tier="cpu",
    description="read per-session transcript CSVs into one zstd parquet (once)",
)
def consolidate(force: bool = False, subsample: int | None = None) -> dict[str, Any]:
    """Consolidate transcripts into ``data/interim/transcripts.parquet``.

    ``subsample`` limits the number of sessions (used by selftest). The full run is a
    once-only CPU job intended for a High-RAM runtime.
    """
    stage_local()  # ensure transcripts extracted locally (no-op if already)
    out = consolidated_path()
    if is_cached(out) and not force and subsample is None:
        log.info("consolidate: cache hit %s (force=True to redo)", out)
        prev = pd.read_parquet(out, columns=["session_id"])
        return {
            "output_path": str(out),
            "n_sessions": int(prev["session_id"].nunique()),
            "n_utterances": int(len(prev)),
            "cached": True,
        }

    tdir = transcripts_dir()
    files = list(iter_files(tdir, "*.csv"))
    if subsample is not None:
        selected = set(subsample_session_ids(subsample))
        files = [path for path in files if path.stem in selected]
    if not files:
        raise FileNotFoundError(f"no transcript CSVs under {tdir} (did staging run?)")

    frames: list[pd.DataFrame] = []
    bad = 0
    for path in pbar(files, desc="consolidate transcripts", unit="session"):
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:  # keep going; report count
            bad += 1
            log.debug("consolidate: unreadable %s (%s)", path.name, exc)
            continue
        if "session_id" not in df.columns:
            df["session_id"] = path.stem
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = _normalize_transcript_frame(full)

    # Subsampled runs (selftest) write to a distinct path to avoid clobbering the
    # real consolidated cache.
    dest = out if subsample is None else interim_dir() / f"transcripts_sub{subsample}.parquet"
    write_parquet(full, dest)

    result = {
        "output_path": str(dest),
        "n_sessions": int(full["session_id"].nunique()),
        "n_utterances": int(len(full)),
        "n_unreadable_files": bad,
        "roles": sorted(full["role"].dropna().unique().tolist())[:20],
        "cached": False,
    }
    log.info(
        "consolidate: %d sessions, %d utterances -> %s",
        result["n_sessions"],
        result["n_utterances"],
        dest,
    )
    return result


def _normalize_transcript_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns and add derived fields (elapsed seconds, order)."""
    expected = ["session_id", "utterance_id", "role", "content", "timestamp"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    df["role"] = df["role"].astype("string").str.strip().str.lower()
    df["content"] = df["content"].astype("string")
    # numeric utterance order (utterance_id is unique-within-session, usually 0..n)
    df["utterance_idx"] = pd.to_numeric(df["utterance_id"], errors="coerce")
    df["t_seconds"] = df["timestamp"].map(parse_elapsed_seconds).astype("float32")
    # stable order within a session: by utterance_idx if present, else by time
    df = df.sort_values(
        ["session_id", "utterance_idx", "t_seconds"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    keep = expected + ["utterance_idx", "t_seconds"]
    return df[keep]
