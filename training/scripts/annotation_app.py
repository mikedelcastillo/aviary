#!/usr/bin/env python3
"""`uv run annotation` — launch the Next.js bird-annotation web tool on :5000.

Ports the old scripts/annotation.sh: the filesystem is the database (annotations
written next to the raw images under data/annotation/raw, labels from
training/roster.yaml). Run it from the repo root.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path.cwd()
    ann = repo / "annotation"
    if not ann.is_dir():
        sys.exit(f"annotation/ not found under {repo} — run `uv run annotation` from the repo root.")

    env = {
        **os.environ,
        # Absolute paths so the Next.js server resolves them regardless of cwd.
        "AVIARY_DATA_ROOT": str(repo / "data" / "annotation" / "raw"),
        "AVIARY_ROSTER": str(repo / "training" / "roster.yaml"),
    }

    if not (ann / "node_modules").is_dir():
        print("Installing annotation tool dependencies (first run)...")
        subprocess.run(["npm", "install"], cwd=ann, env=env, check=True)

    print("Building annotation tool...")
    subprocess.run(["npm", "run", "build"], cwd=ann, env=env, check=True)

    print("Annotation tool -> http://0.0.0.0:5000")
    raise SystemExit(subprocess.run(["npm", "run", "start"], cwd=ann, env=env).returncode)


if __name__ == "__main__":
    main()
