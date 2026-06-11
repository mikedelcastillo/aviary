"""Versioned model-output paths.

Trained weights are exported to ``data/models/`` as ``<name>-NNN.pt`` (e.g.
``live-001.pt``, ``archive-001.pt``). Each export increments the number rather
than overwriting, so prior weights stay around for comparison and rollback.
"""

from __future__ import annotations

import re
from pathlib import Path


def next_version_path(export_dir: Path, name: str) -> Path:
    """Return the next incrementing path ``<export_dir>/<name>-NNN.pt``.

    Scans ``export_dir`` for existing ``<name>-NNN.pt`` files and returns one
    past the highest number (``001`` if none exist). The numeric field is
    zero-padded to at least three digits and grows past 999 naturally.
    """
    pattern = re.compile(rf"^{re.escape(name)}-(\d+)\.pt$")
    highest = 0
    if export_dir.exists():
        for child in export_dir.iterdir():
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return export_dir / f"{name}-{highest + 1:03d}.pt"
