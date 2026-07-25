"""Portable pre-commit guard mirroring .git/hooks/pre-commit.

Blocks any staged file that (a) matches a competition-data pattern or
(b) exceeds 1 MB. Used by the pre-commit framework and CI; the native
git hook is the always-on backstop.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_BYTES = 1_048_576
DATA_RE = re.compile(
    r"(^|/)(train_|test_|submission_format)"
    r"|\.(zip|csv|parquet|pt|bin|safetensors|pth|onnx|joblib|pkl)$"
    r"|(^|/)(data|artifacts|runs|submission)/"
)


def main(argv: list[str]) -> int:
    failed = False
    for name in argv:
        if DATA_RE.search(name):
            print(f"BLOCKED (data pattern): {name}", file=sys.stderr)
            failed = True
            continue
        p = Path(name)
        if p.is_file() and p.stat().st_size > MAX_BYTES:
            print(
                f"BLOCKED (>1MB: {p.stat().st_size} bytes): {name}",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
