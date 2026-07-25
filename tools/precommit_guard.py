"""Portable pre-commit guard mirroring .git/hooks/pre-commit.

Blocks any staged file that (a) matches a competition-data pattern or (b) exceeds 1 MB.
Used by the pre-commit framework and CI; the native git hook is the always-on backstop.

**Patterns are scoped to data extensions on purpose.** An earlier version used a bare
``test_`` prefix, which also matched ``tests/test_*.py`` and silently kept the entire test
suite out of the repository — CI then collected zero tests and failed, while every gate
passed locally.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_BYTES = 1_048_576

# 1. any file carrying a data/model extension
EXT_RE = re.compile(r"\.(zip|csv|parquet|pt|bin|safetensors|pth|onnx|joblib|pkl)$", re.I)
# 2. competition-data filenames, scoped to data extensions
NAME_RE = re.compile(
    r"(^|/)(train_|test_)[^/]*\.(csv|zip|parquet)$"
    r"|(^|/)submission_format[^/]*\.csv$",
    re.I,
)
# 3. anything inside a data/output directory
DIR_RE = re.compile(r"(^|/)(data|artifacts|runs|submission)/")


def is_blocked(name: str) -> bool:
    return bool(EXT_RE.search(name) or NAME_RE.search(name) or DIR_RE.search(name))


def main(argv: list[str]) -> int:
    failed = False
    for name in argv:
        if is_blocked(name):
            print(f"BLOCKED (data pattern): {name}", file=sys.stderr)
            failed = True
            continue
        p = Path(name)
        if p.is_file() and p.stat().st_size > MAX_BYTES:
            print(f"BLOCKED (>1MB: {p.stat().st_size} bytes): {name}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
