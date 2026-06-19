#!/usr/bin/env python3
"""Suggest a roster identity for each box on annotation images.

Runs the custom ``live`` model and, for every box present — human-drawn boxes in
``<image>.json`` AND unapproved box proposals in ``<image>.suggest.json`` — finds
the best-overlapping detection (by IoU) and records its label as a suggestion.

Results land in the suggestion sidecar (see aviary_training.suggest):
  - label suggestions for human ``.json`` boxes go in ``labels`` (keyed by box id);
  - matched box proposals get their ``label``/``labelConf`` filled in.

In the annotation tool's Label mode the suggested pill is pre-highlighted for
one-key approval. The model's class names are roster names, so they map straight
through. Geometry and box proposals are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aviary_training.suggest import (
    iter_images,
    latest_model,
    match_box,
    read_annotation_boxes,
    read_suggestions,
    write_suggestions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/annotation/raw"),
        help="Directory of annotation images to scan (recursive)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO weights. Default: the latest data/models/live-NNN.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Min IoU to match a detection to an existing box",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/models"),
        help="Where to look for the latest live model when --model is omitted",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. cpu, cuda:0, mps). Default: ultralytics auto",
    )
    return parser.parse_args()


def resolve_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    latest = latest_model(args.models_dir.resolve(), "live")
    if latest is None:
        raise SystemExit(
            f"No live-NNN.pt found under {args.models_dir.resolve()}. "
            "Train one (scripts/train_live) or pass --model."
        )
    return str(latest)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    model_path = resolve_model(args)

    from ultralytics import YOLO

    model = YOLO(model_path)
    names = model.names

    def class_name(cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        return str(names[cls_id]) if cls_id < len(names) else str(cls_id)

    predict_args = {"conf": args.conf, "verbose": False}
    if args.device is not None:
        predict_args["device"] = args.device

    images = list(iter_images(source))
    labeled = 0
    total_suggestions = 0

    for image_path in images:
        human_boxes = read_annotation_boxes(image_path)
        sugg = read_suggestions(image_path)
        sugg_boxes = sugg["boxes"]

        # One combined candidate list: human boxes first, then box proposals. The
        # split index lets us route each match back to the right sidecar field.
        candidates = human_boxes + sugg_boxes
        if not candidates:
            continue

        # Best (label, conf) assigned to each candidate; highest-conf detection wins.
        best: dict[int, tuple[str, float]] = {}

        results = model.predict(source=str(image_path), **predict_args)
        if results:
            for det in results[0].boxes:
                cx, cy, w, h = det.xywhn[0].tolist()
                idx = match_box({"cx": cx, "cy": cy, "w": w, "h": h}, candidates, args.iou)
                if idx is None:
                    continue
                conf = float(det.conf[0].item())
                if idx not in best or conf > best[idx][1]:
                    best[idx] = (class_name(int(det.cls[0].item())), conf)

        if not best:
            continue

        split = len(human_boxes)
        labels = []
        for idx, (label, conf) in best.items():
            if idx < split:  # a human .json box -> keyed label suggestion
                labels.append({"boxId": human_boxes[idx]["id"], "label": label, "conf": round(conf, 4)})
            else:  # a box proposal -> fill its own label fields
                sugg_boxes[idx - split]["label"] = label
                sugg_boxes[idx - split]["labelConf"] = round(conf, 4)

        sugg["labels"] = labels
        sugg["labelModel"] = Path(model_path).name
        write_suggestions(image_path, sugg)

        labeled += 1
        total_suggestions += len(best)
        print(f"{image_path.name}: {len(best)} label suggestion(s)")

    print(
        f"Done. {total_suggestions} label suggestions across {labeled} image(s) "
        f"using {Path(model_path).name}."
    )


if __name__ == "__main__":
    main()
