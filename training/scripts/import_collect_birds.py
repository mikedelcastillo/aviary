#!/usr/bin/env python3
"""Import auto-collected bird frames into the annotation pipeline.

The server sorts captures into per-class folders under `data/server/collect/`
(e.g. `bird/`, `draft/`, `matcha/`). This imports the generic `bird` folder PLUS
every roster-named folder (`training/roster.yaml` is the single source of truth),
skipping any that don't exist yet. Non-roster folders (cat/, dog/, …) are ignored.

For each folder it reads `*.jpg` + paired `*.json` detection files, classifies
each frame as `day` (color) or `ir` (infrared/night) from image content, moves it
into `data/annotation/raw/tapo/{day,ir}/` with a clean name, and writes a
non-authoritative `<stem>.suggest.json` from the detection box so the bird shows
up as a YELLOW suggestion (accept/reject) when you open the frame to label it. The
folder name carries the roster identity (the generic `bird` folder stays
unlabeled). The consumed detection `.json` is deleted.

Day/IR is decided from content, not timestamps: IR/night frames are grayscale
(R=G=B) even though the JPEG is still 3-channel, so we measure the fraction of
pixels with chroma (max-min across channels) above a small threshold. Color
frames have many such pixels; IR frames have ~none.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from aviary_training.roster import load_roster
from aviary_training.suggest import write_suggestions

console = Console()

# YYYYMMDD_HHMMSS_microseconds_camera-N_bird_HASH
STEM_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_\d+_(?P<camera>camera-\d+)_\w+_(?P<hash>[0-9a-f]+)$"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-root",
        type=Path,
        default=repo_root / "data" / "server" / "collect",
        help="Parent folder holding the per-class collection subfolders",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=repo_root / "training" / "roster.yaml",
        help="Roster YAML naming the per-class folders to import",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "annotation" / "raw" / "tapo",
        help="Destination with day/ and ir/ subfolders",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4)),
        help="Parallel image workers (default: min(16, cpu count))",
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
        help="Skip suggestion sidecar generation (sort/move images only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print the plan; move/write/delete nothing",
    )
    return parser.parse_args()


def collect_folders(roster_path: Path) -> list[str]:
    """"bird" + every roster label name, de-duplicated, order preserved."""
    names = ["bird"] + [label.name for label in load_roster(roster_path)]
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


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


def suggestion_box(detection: dict, label: str | None) -> dict | None:
    """Build ONE non-authoritative proposal box from a detection's pixel bbox.

    Returns a box dict shaped for ``<image>.suggest.json`` — normalized
    center+size geometry plus ``conf``/``label``/``labelConf`` — that the
    annotation tool renders as a YELLOW suggestion to accept or reject. These are
    proposals, not ground truth: dataset prep excludes any frame without a
    human-confirmed ``.json`` (``aviary_training.annotations``), so they never
    reach training until a human reviews the frame.

    ``label`` is the folder's roster identity, or None for the generic ``bird``
    folder (proposal stays unlabeled). Returns None when the detection lacks the
    geometry needed to place a box.
    """
    frame = detection.get("frame", {})
    bbox = detection.get("detection", {}).get("bbox_xyxy")
    width = frame.get("width")
    height = frame.get("height")
    if not bbox or not width or not height:
        return None

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    clamp = lambda v: min(1.0, max(0.0, v))
    cx = clamp(((x1 + x2) / 2) / width)
    cy = clamp(((y1 + y2) / 2) / height)
    w = clamp((x2 - x1) / width)
    h = clamp((y2 - y1) / height)

    raw_conf = detection.get("detection", {}).get("confidence")
    conf = float(raw_conf) if raw_conf is not None else 1.0

    return {
        "id": "sug-0",
        "cx": round(cx, 6),
        "cy": round(cy, 6),
        "w": round(w, 6),
        "h": round(h, 6),
        "conf": round(conf, 4),
        "label": label,
        "labelConf": round(conf, 4) if label is not None else None,
    }


def classify_and_apply(
    image_path: Path, args, day_dir: Path, ir_dir: Path, class_name: str
) -> dict:
    """Classify one frame day/ir, then move + suggest-box it (or plan it under dry-run).

    ``class_name`` is the collection folder's roster identity; the generic
    ``bird`` folder maps to an UNLABELED proposal (label None), every other folder
    stamps its name onto the suggestion.

    Returns a record `{lighting, label, skipped, msg}` (``label`` now means "a
    suggestion sidecar was written"). Self-contained and side-effect-isolated
    (distinct source/dest per image), so it is safe to run in a thread pool; the
    caller folds the record into shared totals on the main thread. `msg` carries
    an optional rich-markup line for the caller to print.
    """
    from PIL import Image

    result = {"lighting": None, "label": False, "skipped": False, "msg": None}

    try:
        with Image.open(image_path) as img:
            img.draft("RGB", (288, 162))  # decoder downscales ~1/8; same day/ir result
            frac = colored_fraction(img.convert("RGB"))
    except Exception as exc:  # unreadable/corrupt image
        result["skipped"] = True
        result["msg"] = f"[red]✗[/] (read error) {image_path.name}: {exc}"
        return result

    lighting = "ir" if frac < args.ir_threshold else "day"
    dest_dir = ir_dir if lighting == "ir" else day_dir

    json_path = image_path.with_suffix(".json")
    detection: dict | None = None
    if json_path.exists():
        try:
            detection = json.loads(json_path.read_text())
        except Exception as exc:
            result["msg"] = f"[yellow]⚠[/] (bad json) {json_path.name}: {exc}"

    stem = target_stem(image_path.stem, lighting, detection)
    dest_jpg = dest_dir / f"{stem}.jpg"

    if dest_jpg.exists():
        result["skipped"] = True
        return result

    # The generic "bird" folder has no identity -> leave the proposal unlabeled.
    label = None if class_name == "bird" else class_name

    sug_box = None
    if not args.no_labels and detection is not None:
        sug_box = suggestion_box(detection, label)
        if sug_box is None:
            result["msg"] = (
                f"[yellow]⚠[/] (no bbox) {json_path.name}: image moved without suggestion"
            )

    if args.dry_run:
        extra = " [green]+suggestion[/]" if sug_box else ""
        result["msg"] = (
            f"[dim][dry-run][/] → {lighting}/{dest_jpg.name}{extra} (frac={frac:.4f})"
        )
        result["lighting"] = lighting
        result["label"] = sug_box is not None
        return result

    image_path.rename(dest_jpg)
    if sug_box:
        write_suggestions(dest_jpg, {"boxes": [sug_box], "labels": []})
        result["label"] = True
    if json_path.exists():
        json_path.unlink()

    result["lighting"] = lighting
    return result


def import_folder(source: Path, args, day_dir: Path, ir_dir: Path, totals: dict) -> None:
    """Sort/move/suggest-box every *.jpg in one collection folder, updating `totals`."""
    images = sorted(source.glob("*.jpg"))
    if not images:
        console.print(f"[dim]·[/] [cyan]{source.name}[/]: no images")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{source.name}[/] · {len(images)} images", total=len(images))

        # The folder name IS the roster identity for these captures.
        class_name = source.name
        worker = lambda image_path: classify_and_apply(
            image_path, args, day_dir, ir_dir, class_name
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # map preserves submission order, so dry-run output stays ordered.
            for record in executor.map(worker, images):
                if record["msg"]:
                    progress.console.print(record["msg"])
                if record["skipped"]:
                    totals["skipped"] += 1
                elif record["lighting"]:
                    totals[record["lighting"]] += 1
                    if record["label"]:
                        totals["suggestions"] += 1
                progress.advance(task)


def main() -> None:
    args = parse_args()

    if not args.collect_root.is_dir():
        raise SystemExit(f"Collection root not found: {args.collect_root}")

    folders = [args.collect_root / name for name in collect_folders(args.roster)]
    present = [f for f in folders if f.is_dir()]

    day_dir = args.output / "day"
    ir_dir = args.output / "ir"
    if not args.dry_run:
        day_dir.mkdir(parents=True, exist_ok=True)
        ir_dir.mkdir(parents=True, exist_ok=True)

    mode = " [yellow](dry run)[/]" if args.dry_run else ""
    console.print(
        f"[bold]Collect import[/] · [cyan]{len(present)}[/]/{len(folders)} folders present{mode}"
    )
    console.print(f"[dim]{args.collect_root} → {args.output}[/]\n")

    totals = {"day": 0, "ir": 0, "suggestions": 0, "skipped": 0}
    for source in present:
        import_folder(source, args, day_dir, ir_dir, totals)

    verb = "Would import" if args.dry_run else "Imported"
    console.print(
        f"\n[bold green]✓[/] {verb} [green]{totals['day']}[/] day + [green]{totals['ir']}[/] ir, "
        f"[green]{totals['suggestions']}[/] suggestions, "
        f"[yellow]{totals['skipped']}[/] skipped across {len(present)} folder(s)."
    )


if __name__ == "__main__":
    main()
