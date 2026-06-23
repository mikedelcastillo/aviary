#!/usr/bin/env python3
"""`uv run doctor-fsdb [dataRoot]` — FS-as-DB integrity checker.

Thin wrapper around the dependency-free node implementation at
scripts/doctor-fsdb.mjs (kept as-is); run from the repo root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path.cwd() / "scripts" / "doctor-fsdb.mjs"
    if not script.is_file():
        sys.exit(f"{script} not found — run `uv run doctor-fsdb` from the repo root.")
    raise SystemExit(subprocess.run(["node", str(script), *sys.argv[1:]]).returncode)


if __name__ == "__main__":
    main()
