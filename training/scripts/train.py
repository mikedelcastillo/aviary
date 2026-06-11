#!/usr/bin/env python3
"""Train an Ultralytics YOLO detector and optionally export best.pt."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aviary_training.models import next_version_path


def batch_size(value: str) -> int | float:
    """Parse --batch: 'auto' -> -1 (Ultralytics AutoBatch, ~60% GPU memory),
    an int (fixed batch), or a float in (0, 1) (fraction of GPU memory)."""
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--batch",
        type=batch_size,
        default="auto",
        help="int (fixed), float in (0,1) (fraction of GPU memory), or 'auto' (AutoBatch)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, mps")
    parser.add_argument("--project", type=Path, default=Path("data/training/runs"))
    parser.add_argument("--name", default="bird_detector")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to export the trained best.pt into (see --export-name)",
    )
    parser.add_argument(
        "--export-name",
        help="Export best.pt as <export-dir>/<export-name>-NNN.pt, incrementing "
        "NNN each run (e.g. live -> live-001.pt, live-002.pt). Skips export if unset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        # Absolute so Ultralytics writes to data/training/runs/<name> verbatim;
        # a relative project gets nested under its global runs dir (runs/detect/...).
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device != "auto":
        train_args["device"] = args.device

    model.train(**train_args)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished, but best checkpoint was not found at {best}")

    print(f"Best checkpoint: {best}")

    if args.export_name:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        export_to = next_version_path(args.export_dir, args.export_name)
        shutil.copy2(best, export_to)
        print(f"Exported checkpoint to {export_to}")


if __name__ == "__main__":
    main()
