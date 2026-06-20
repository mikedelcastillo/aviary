#!/usr/bin/env python3
"""Train an Ultralytics YOLO detector and optionally export best.pt."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def batch_size(value: str) -> int | float:
    """Parse --batch: 'auto' -> -1 (Ultralytics AutoBatch, ~60% GPU memory),
    an int (fixed batch), or a float in (0, 1) (fraction of GPU memory)."""
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--export-to", type=Path, help="Copy best.pt to this path")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early-stop after this many epochs without val improvement (Ultralytics default 100)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Cap training at this many HOURS (overrides --epochs when set)",
    )
    parser.add_argument(
        "--cls-pw",
        type=float,
        default=None,
        help="Class-weights power for imbalance (Ultralytics cls_pw; 0=off, 1=full inverse "
        "frequency). Up-weights rare classes such as the IR species and unknown_bird.",
    )
    parser.add_argument(
        "--hsv-h",
        type=float,
        default=None,
        help="Hue augmentation gain (Ultralytics hsv_h; default 0.015). Set 0 to never rotate "
        "hue, preserving the color cues that separate same-shape birds.",
    )
    return parser.parse_args(argv)


def build_train_args(args: argparse.Namespace) -> dict:
    """Assemble the Ultralytics ``model.train`` kwargs from parsed CLI args.

    Optional knobs (time, cls_pw, hsv_h, explicit device) are included ONLY when
    set, so unset ones keep Ultralytics' defaults. Pure (no ultralytics import) so
    it is unit-testable without a GPU.
    """
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
        "patience": args.patience,
    }
    if args.time is not None:
        train_args["time"] = args.time
    if args.cls_pw is not None:
        train_args["cls_pw"] = args.cls_pw
    if args.hsv_h is not None:
        train_args["hsv_h"] = args.hsv_h
    if args.device != "auto":
        train_args["device"] = args.device
    return train_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**build_train_args(args))

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished, but best checkpoint was not found at {best}")

    print(f"Best checkpoint: {best}")

    if args.export_to:
        args.export_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.export_to)
        print(f"Exported checkpoint to {args.export_to}")


if __name__ == "__main__":
    main()
