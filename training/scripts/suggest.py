#!/usr/bin/env python3
"""Propose boxes AND roster labels for annotation images in one pass per image.

This merges the old ``suggest_boxes`` + ``suggest_labels`` into a single script so
each image is read/decoded once and the two models' output is combined and
de-duplicated before it is written.

Per image we run two detectors and write their combined output to the
non-authoritative ``<image>.suggest.json`` sidecar (see aviary_training.suggest),
which the annotation tool renders as YELLOW proposals to accept/reject:

  (a) a GENERIC detector (default ``yolo11n.pt``) keeping only its COCO ``bird``
      class -> bird box proposals with ``label=None`` (catches any bird regardless
      of identity);
  (b) the CUSTOM live model (latest ``data/models/live-NNN.pt``, or ``--live-model``)
      -> box proposals carrying roster identities AND label suggestions for
      EXISTING human ``.json`` boxes (matched by IoU);
  (c) all proposed boxes are combined and de-duplicated (dedupe_suggestions),
      dropping generic 'bird'/unlabeled boxes that overlap a specific-label box;
  (d) the combined result is written via write_suggestions — ``boxes`` (with
      labels where the live model knew them) + ``labels`` (for human boxes), plus
      ``boxModel`` / ``labelModel`` provenance.

``--skip-boxed`` (default on) skips images a human already confirmed
(``boxed=true``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aviary_training.suggest import (
    dedupe_suggestions,
    is_boxed,
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
        default="yolo11n.pt",
        help="Generic detector weights (COCO 'bird'). Default: yolo11n.pt (auto-downloaded)",
    )
    parser.add_argument(
        "--live-model",
        default=None,
        help="Custom live weights. Default: the latest data/models/live-NNN.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Min IoU to match a live detection to an existing human box",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/models"),
        help="Where to look for the latest live model when --live-model is omitted",
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


def resolve_live_model(args: argparse.Namespace) -> str | None:
    """Resolve the custom model path, or None when none is available.

    Unlike the old suggest_labels, a missing live model is NOT fatal: the generic
    detector still produces bird box proposals. We only warn.
    """
    if args.live_model:
        return args.live_model
    latest = latest_model(args.models_dir.resolve(), "live")
    return str(latest) if latest else None


def class_namer(model):
    """A ``cls_id -> name`` lookup that tolerates dict- or list-shaped names."""
    names = model.names

    def class_name(cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        return str(names[cls_id]) if cls_id < len(names) else str(cls_id)

    return class_name


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    from ultralytics import YOLO

    generic_model = YOLO(args.model)
    generic_name = class_namer(generic_model)

    live_path = resolve_live_model(args)
    live_model = YOLO(live_path) if live_path else None
    live_name = class_namer(live_model) if live_model else None
    if live_model is None:
        print(
            "No live model found (pass --live-model or train one); "
            "only generic bird boxes will be proposed."
        )

    predict_args = {"conf": args.conf, "verbose": False}
    if args.device is not None:
        predict_args["device"] = args.device

    images = list(iter_images(source))
    total_boxes = 0
    total_labels = 0
    written = 0
    skipped = 0

    for image_path in images:
        if args.skip_boxed and is_boxed(image_path):
            skipped += 1
            continue

        proposals: list[dict] = []

        # (a) Generic detector -> bird boxes, no identity.
        results = generic_model.predict(source=str(image_path), **predict_args)
        if results:
            for det in results[0].boxes:
                if generic_name(int(det.cls[0].item())).lower() != "bird":
                    continue
                cx, cy, w, h = (round(v, 6) for v in det.xywhn[0].tolist())
                proposals.append(
                    {
                        "id": "",  # assigned after dedup
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "conf": round(float(det.conf[0].item()), 4),
                        "label": None,
                        "labelConf": None,
                    }
                )

        # (b) Live model -> identity boxes + label suggestions for human boxes.
        human_boxes = read_annotation_boxes(image_path)
        # best[idx] = (label, conf) for each human box index (highest conf wins).
        best_human: dict[int, tuple[str, float]] = {}
        if live_model is not None:
            live_results = live_model.predict(source=str(image_path), **predict_args)
            if live_results:
                for det in live_results[0].boxes:
                    cx, cy, w, h = (round(v, 6) for v in det.xywhn[0].tolist())
                    conf = round(float(det.conf[0].item()), 4)
                    label = live_name(int(det.cls[0].item()))

                    # New identity box proposal.
                    proposals.append(
                        {
                            "id": "",
                            "cx": cx,
                            "cy": cy,
                            "w": w,
                            "h": h,
                            "conf": conf,
                            "label": label,
                            "labelConf": conf,
                        }
                    )

                    # Also suggest a label for any overlapping human .json box.
                    if human_boxes:
                        idx = match_box(
                            {"cx": cx, "cy": cy, "w": w, "h": h}, human_boxes, args.iou
                        )
                        if idx is not None and (
                            idx not in best_human or conf > best_human[idx][1]
                        ):
                            best_human[idx] = (label, conf)

        # (c) Combine + dedup, then assign stable ids in final order.
        deduped = dedupe_suggestions(proposals)
        for i, b in enumerate(deduped):
            b["id"] = f"sug-{i}"

        labels = [
            {"boxId": human_boxes[idx]["id"], "label": label, "conf": conf}
            for idx, (label, conf) in sorted(best_human.items())
        ]

        # (d) Write combined sidecar with provenance.
        data = read_suggestions(image_path)
        data["boxes"] = deduped
        data["labels"] = labels
        data["boxModel"] = Path(args.model).name
        if live_path:
            data["labelModel"] = Path(live_path).name
        write_suggestions(image_path, data)

        total_boxes += len(deduped)
        total_labels += len(labels)
        written += 1
        if deduped or labels:
            print(
                f"{image_path.name}: {len(deduped)} box + {len(labels)} label suggestion(s)"
            )

    live_note = Path(live_path).name if live_path else "no live model"
    print(
        f"Done. {total_boxes} box + {total_labels} label suggestions across "
        f"{written} image(s); {skipped} already-boxed image(s) skipped "
        f"(generic={Path(args.model).name}, live={live_note})."
    )


if __name__ == "__main__":
    main()
