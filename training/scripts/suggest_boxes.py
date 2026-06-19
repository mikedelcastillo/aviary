#!/usr/bin/env python3
"""Propose bird bounding boxes for annotation images using a YOLO model.

Writes proposals to ``<image>.suggest.json`` (the non-authoritative suggestion
sidecar — see aviary_training.suggest). In the annotation tool's Box mode these
render as yellow boxes the user approves (becomes a real box) or rejects.

Defaults to a generic pretrained ``yolo11n.pt`` and keeps only its COCO ``bird``
class, so it catches any bird regardless of identity. Point ``--model`` at a
custom model and pass ``--class-name any`` to keep every detected class instead.
``suggest_labels`` assigns identities afterward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aviary_training.suggest import (
    is_boxed,
    iter_images,
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
        default="yolo11n.pt",
        help="YOLO weights. Default: generic pretrained yolo11n.pt (auto-downloaded)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--class-name",
        default="bird",
        help="Keep only detections of this class name; use 'any' to keep all classes",
    )
    parser.add_argument(
        "--skip-boxed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images a human has already confirmed (boxed=true). Default: on",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. cpu, cuda:0, mps). Default: ultralytics auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    keep_all = args.class_name.lower() in ("", "any", "*", "all")
    want = args.class_name.lower()

    from ultralytics import YOLO

    model = YOLO(args.model)
    names = model.names  # dict[int, str] (or list) of class names

    def class_name(cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        return str(names[cls_id]) if cls_id < len(names) else str(cls_id)

    predict_args = {"conf": args.conf, "verbose": False}
    if args.device is not None:
        predict_args["device"] = args.device

    images = list(iter_images(source))
    total_boxes = 0
    written = 0
    skipped = 0

    for image_path in images:
        if args.skip_boxed and is_boxed(image_path):
            skipped += 1
            continue

        results = model.predict(source=str(image_path), **predict_args)
        proposals = []
        if results:
            for det in results[0].boxes:
                cls_id = int(det.cls[0].item())
                if not keep_all and class_name(cls_id).lower() != want:
                    continue
                cx, cy, w, h = (round(v, 6) for v in det.xywhn[0].tolist())
                proposals.append(
                    {
                        "id": f"sug-{len(proposals)}",
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "conf": round(float(det.conf[0].item()), 4),
                        "label": None,
                        "labelConf": None,
                    }
                )

        # Preserve any label suggestions already written by suggest_labels;
        # overwrite only the box proposals + provenance.
        data = read_suggestions(image_path)
        data["boxes"] = proposals
        data["boxModel"] = Path(args.model).name
        write_suggestions(image_path, data)

        total_boxes += len(proposals)
        written += 1
        if proposals:
            print(f"{image_path.name}: {len(proposals)} box suggestion(s)")

    print(
        f"Done. {total_boxes} box suggestions across {written} image(s); "
        f"{skipped} already-boxed image(s) skipped."
    )


if __name__ == "__main__":
    main()
