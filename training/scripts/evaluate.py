#!/usr/bin/env python3
"""Evaluate a trained YOLO model against an Ultralytics dataset.

Prints the Ultralytics val metrics AND appends a compact record (mAP50,
mAP50-95, precision/recall, per-stage speed) to ``data/models/evaluate.json``,
so the standard-metric view accumulates history next to benchmark.json instead
of vanishing into stdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/evaluate.json"),
        help="JSON history file the metrics record is appended to",
    )
    return parser.parse_args()


def summarize_val(metrics) -> dict:
    """The val metrics worth keeping, from an Ultralytics results object.

    Tolerant of missing attributes (pure so it is unit-testable without
    ultralytics): absent values record as None rather than crashing the run
    that just spent minutes evaluating.
    """
    box = getattr(metrics, "box", None)

    def _metric(name: str):
        value = getattr(box, name, None)
        try:
            return round(float(value), 4) if value is not None else None
        except (TypeError, ValueError):
            return None

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        speed = {key: round(float(value), 2) for key, value in speed.items()}
    else:
        speed = None
    return {
        "map50": _metric("map50"),
        "map50_95": _metric("map"),
        "precision": _metric("mp"),
        "recall": _metric("mr"),
        "speed_ms": speed,
    }


def main() -> None:
    args = parse_args()

    from bench_speed import append_record, git_revision
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

    record = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
        "model": str(args.model),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        **summarize_val(metrics),
    }
    append_record(args.output, record)
    print(f"recorded -> {args.output}")


if __name__ == "__main__":
    main()
