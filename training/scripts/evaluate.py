#!/usr/bin/env python3
"""Evaluate a trained YOLO model against an Ultralytics dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    val_args = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "split": args.split,
    }
    if args.device != "auto":
        val_args["device"] = args.device

    metrics = model.val(**val_args)
    print(metrics)


if __name__ == "__main__":
    main()
