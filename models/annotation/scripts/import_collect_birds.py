#!/usr/bin/env python3
"""Import auto-collected bird frames into the annotation pipeline.

Reads `*.jpg` + paired `*.json` detection files from a collection folder
(default `collect/bird`), classifies each frame as `day` (color) or `ir`
(infrared/night) from image content, moves it into
`models/annotation/raw/camera_frames/{day,ir}/` with a clean name, and writes
a YOLO sidecar `.txt` from the detection box so the bird is pre-boxed when you
open the frame to label it. The consumed `.json` is deleted.

Day/IR is decided from content, not timestamps: IR/night frames are grayscale
(R=G=B) even though the JPEG is still 3-channel, so we measure the fraction of
pixels with chroma (max-min across channels) above a small threshold. Color
frames have many such pixels; IR frames have ~none.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# YYYYMMDD_HHMMSS_microseconds_camera-N_bird_HASH
STEM_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_\d+_(?P<camera>camera-\d+)_\w+_(?P<hash>[0-9a-f]+)$"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "collect" / "bird",
        help="Folder of *.jpg + *.json pairs to import",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "models" / "annotation" / "raw" / "camera_frames",
        help="Destination with day/ and ir/ subfolders",
    )
    parser.add_argument(
        "--ir-threshold",
        type=float,
        default=0.01,
        help="colored_frac below this is classified as ir (default 0.01)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip YOLO sidecar generation (sort/move images only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print the plan; move/write/delete nothing",
    )
    return parser.parse_args()


def colored_fraction(image) -> float:
    """Fraction of pixels with chroma (max-min across RGB) above 20.

    ~0 for grayscale IR frames, well above the threshold for color frames.
    """
    import numpy as np

    a = np.asarray(image, dtype=np.int16)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return float((mx - mn > 20).mean())


def target_stem(source_stem: str, lighting: str, detection: dict | None) -> str:
    """Build `{camera}_{lighting}_{YYYY-MM-DD}_{HHMMSS}_{hash}`.

    Falls back to the detection JSON / original stem when the filename does not
    match the expected pattern.
    """
    match = STEM_RE.match(source_stem)
    if match:
        date = match["date"]
        date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        return f"{match['camera']}_{lighting}_{date_fmt}_{match['time']}_{match['hash']}"

    # Fallback: derive what we can from the JSON, keep the original stem for uniqueness.
    camera = "camera"
    if detection:
        camera = str(detection.get("camera", {}).get("name", "camera"))
    return f"{camera}_{lighting}_{source_stem}"


def yolo_line(detection: dict) -> str | None:
    """Convert a detection's pixel bbox_xyxy to a normalized YOLO line (class 0)."""
    frame = detection.get("frame", {})
    bbox = detection.get("detection", {}).get("bbox_xyxy")
    width = frame.get("width")
    height = frame.get("height")
    if not bbox or not width or not height:
        return None

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    clamp = lambda v: min(1.0, max(0.0, v))
    return f"0 {clamp(cx):.6f} {clamp(cy):.6f} {clamp(w):.6f} {clamp(h):.6f}"


def main() -> None:
    args = parse_args()
    from PIL import Image

    if not args.source.is_dir():
        raise SystemExit(f"Source folder not found: {args.source}")

    day_dir = args.output / "day"
    ir_dir = args.output / "ir"
    if not args.dry_run:
        day_dir.mkdir(parents=True, exist_ok=True)
        ir_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.source.glob("*.jpg"))
    if not images:
        print(f"No *.jpg files in {args.source}")
        return

    counts = {"day": 0, "ir": 0}
    labels_written = 0
    skipped = 0

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                frac = colored_fraction(img.convert("RGB"))
        except Exception as exc:  # unreadable/corrupt image
            print(f"SKIP (read error) {image_path.name}: {exc}")
            skipped += 1
            continue

        lighting = "ir" if frac < args.ir_threshold else "day"
        dest_dir = ir_dir if lighting == "ir" else day_dir

        json_path = image_path.with_suffix(".json")
        detection: dict | None = None
        if json_path.exists():
            try:
                detection = json.loads(json_path.read_text())
            except Exception as exc:
                print(f"WARN (bad json) {json_path.name}: {exc}")

        stem = target_stem(image_path.stem, lighting, detection)
        dest_jpg = dest_dir / f"{stem}.jpg"
        dest_txt = dest_dir / f"{stem}.txt"

        if dest_jpg.exists():
            print(f"SKIP (exists) {image_path.name} -> {lighting}/{dest_jpg.name}")
            skipped += 1
            continue

        line = None
        if not args.no_labels and detection is not None:
            line = yolo_line(detection)
            if line is None:
                print(f"WARN (no bbox) {json_path.name}: image moved without sidecar")

        if args.dry_run:
            extra = " +label" if line else ""
            print(f"[dry-run] {image_path.name} -> {lighting}/{dest_jpg.name}{extra} (frac={frac:.4f})")
            counts[lighting] += 1
            if line:
                labels_written += 1
            continue

        image_path.rename(dest_jpg)
        if line:
            dest_txt.write_text(line + "\n")
            labels_written += 1
        if json_path.exists():
            json_path.unlink()

        counts[lighting] += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"\n{prefix}Imported {counts['day']} day + {counts['ir']} ir frames, "
        f"{labels_written} sidecars, {skipped} skipped."
    )


if __name__ == "__main__":
    main()
