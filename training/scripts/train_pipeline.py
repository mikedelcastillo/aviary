#!/usr/bin/env python3
"""Shared end-to-end build pipeline for the live and archive bird detectors.

Ports the logic that used to live in ``scripts/train_{live,archive}.sh``:
prepare a dataset from the labeled raw images, compute the next zero-padded
``data/models/<model>-NNN.pt`` sequence number, then train and export to it.

The per-model entry points (``train_live.py`` / ``train_archive.py``) are thin
wrappers that call :func:`run` with their model name. We reuse the existing
``prepare_dataset`` and ``train`` ``main()`` functions instead of duplicating the
prepare/train logic.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import prepare_dataset
import train

# data/models/<model>-NNN.pt — the export naming benchmark.py also parses.
_SEQ_RE = re.compile(r"-(\d+)$")


def next_export_path(models_dir: Path, model: str) -> Path:
    """Next ``<models_dir>/<model>-NNN.pt`` after the highest existing sequence.

    Each run is preserved and never clobbers the model the server may currently
    be loading. Numbering is zero-padded to three digits, starting at 001.
    """
    last = 0
    for path in models_dir.glob(f"{model}-*.pt"):
        match = _SEQ_RE.search(path.stem)
        if not match:
            continue
        last = max(last, int(match.group(1)))
    return models_dir / f"{model}-{last + 1:03d}.pt"


def run(model: str, argv: list[str] | None = None) -> None:
    """Build ``model`` (``live`` or ``archive``) end-to-end.

    Extra CLI flags in ``argv`` pass through to training (e.g. ``--epochs 200
    --device cuda:0 --model yolo11s.pt``), matching the old wrapper behavior.
    """
    source = Path(os.environ.get("AVIARY_LABEL_SOURCE", "data/annotation/raw"))
    dataset = Path("data/training/datasets") / model

    # Rebuild the dataset from scratch (prepare_dataset wipes its split folders,
    # but drop the whole dir to mirror the old `rm -rf "$DATASET"`).
    shutil.rmtree(dataset, ignore_errors=True)
    prepare_dataset.main(
        [
            "--source", str(source),
            "--output", str(dataset),
            "--model", model,
        ]
    )

    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    export = next_export_path(models_dir, model)

    train.main(
        [
            "--data", str(dataset / "dataset.yaml"),
            "--name", model,
            "--export-to", str(export),
            *(argv or []),
        ]
    )

    print(f"Exported {model} model to {export}")
