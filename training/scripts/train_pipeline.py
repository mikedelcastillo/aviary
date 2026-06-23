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

import cut_paste_ir
import prepare_dataset
import train

# data/models/<model>-NNN.pt — the export naming benchmark.py also parses.
_SEQ_RE = re.compile(r"-(\d+)$")


def augment_ir(dataset: Path) -> None:
    """Merge species-balanced cut-paste IR composites into the train split.

    The live-018 recipe: budgie/lovebird/cockatiel crops are IR-toned and pasted
    non-overlapping into REAL night-IR backgrounds, then added to the train split
    only (val/test stay 100% real so any later A/B stays honest). Lifts night-IR
    recall — measured +2 overall, with no day-class regression — over a plain
    real-data build. Uses the validated defaults (300 composites, seed 0).
    """
    stage = dataset.parent.parent / "_cutpaste_stage"
    shutil.rmtree(stage, ignore_errors=True)
    cut_paste_ir.main(
        ["--dataset-dir", str(dataset), "--output-dir", str(stage), "--count", "300", "--seed", "0"]
    )

    img_train = dataset / "images" / "train"
    lbl_train = dataset / "labels" / "train"
    for src in sorted((stage / "images").glob("*")):
        shutil.copy2(src, img_train / src.name)
    for src in sorted((stage / "labels").glob("*")):
        shutil.copy2(src, lbl_train / src.name)
    # Drop the scan cache so Ultralytics re-reads the train split with the new files.
    lbl_train.with_suffix(".cache").unlink(missing_ok=True)


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

    Cut-paste IR augmentation (the live-018 method; see :func:`augment_ir`) is ON
    by default; pass ``--no-augment-ir`` to build from real labels only.
    """
    argv = list(argv or [])

    # Handle --help BEFORE any side effects: this command rebuilds the dataset
    # (a destructive re-prep), so `train-<model> --help` must NOT trigger it.
    if "-h" in argv or "--help" in argv:
        print(
            f"{model}: prepare a dataset from the labeled raw images, then train and "
            f"export data/models/{model}-NNN.pt.\n"
            "  --no-augment-ir  build from real labels only; by DEFAULT this command\n"
            "                   injects species-balanced cut-paste IR composites into\n"
            "                   the train split (the live-018 method; lifts night-IR\n"
            "                   recall). Other flags pass through to training:\n"
        )
        train.main(["--help"])  # argparse prints the training options, then exits
        return

    # Cut-paste IR augmentation is ON by default (the live-018 method). Strip the
    # toggle flags here: they're a pipeline step, not training args, and train.py's
    # argparse would reject them. --augment-ir is accepted as an explicit no-op.
    augment = "--no-augment-ir" not in argv
    for flag in ("--augment-ir", "--no-augment-ir"):
        while flag in argv:
            argv.remove(flag)

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

    # Inject composites AFTER the real dataset is built but BEFORE training, so the
    # pasted night-IR frames join the train split (val/test untouched).
    if augment:
        augment_ir(dataset)

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
