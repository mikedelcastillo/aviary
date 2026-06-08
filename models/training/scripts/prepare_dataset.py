#!/usr/bin/env python3
"""Normalize a CVAT YOLO export into an Ultralytics dataset directory."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_NAMES = {
    0: "bird_1",
    1: "bird_2",
    2: "bird_3",
    3: "bird_4",
    4: "bird_5",
    5: "bird_6",
    6: "unknown_bird",
}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="CVAT YOLO export directory")
    parser.add_argument("--output", required=True, type=Path, help="Output Ultralytics dataset directory")
    parser.add_argument("--names", type=Path, default=Path("models/training/config/birds.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-labels",
        action="store_true",
        help="Fail if any image is missing a matching YOLO .txt label file",
    )
    return parser.parse_args()


def load_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return DEFAULT_NAMES

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    names = data.get("names", DEFAULT_NAMES)
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if "__MACOSX" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def index_labels(root: Path) -> dict[str, list[Path]]:
    labels: dict[str, list[Path]] = {}
    for path in root.rglob("*.txt"):
        if "__MACOSX" in path.parts:
            continue
        labels.setdefault(path.stem, []).append(path)
    return labels


def choose_label(image: Path, labels_by_stem: dict[str, list[Path]]) -> Path | None:
    candidates = labels_by_stem.get(image.stem, [])
    if not candidates:
        return None

    for candidate in candidates:
        if candidate.parent == image.parent:
            return candidate

    for candidate in candidates:
        if "label" in {part.lower() for part in candidate.parts}:
            return candidate

    return candidates[0]


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val-ratio and test-ratio must be non-negative and sum to less than 1")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    count = len(shuffled)
    test_count = int(round(count * test_ratio))
    val_count = int(round(count * val_ratio))

    if count >= 3:
        test_count = max(1, test_count)
        val_count = max(1, val_count)

    train_count = count - val_count - test_count
    if train_count <= 0:
        train_count = max(1, count)
        val_count = 0
        test_count = 0

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def slug_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return "__".join(relative.parts).replace(" ", "_")


def copy_split(split: str, samples: list[Sample], source: Path, output: Path) -> list[dict[str, str]]:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for sample in samples:
        dest_stem = slug_for(sample.image, source)
        dest_image = image_dir / f"{dest_stem}{sample.image.suffix.lower()}"
        dest_label = label_dir / f"{dest_stem}.txt"

        shutil.copy2(sample.image, dest_image)
        if sample.label:
            shutil.copy2(sample.label, dest_label)
            label_status = "provided"
        else:
            dest_label.write_text("", encoding="utf-8")
            label_status = "empty"

        rows.append(
            {
                "split": split,
                "source_image": str(sample.image),
                "source_label": str(sample.label or ""),
                "output_image": str(dest_image),
                "output_label": str(dest_label),
                "label_status": label_status,
            }
        )
    return rows


def write_dataset_yaml(output: Path, names: dict[int, str]) -> None:
    import yaml

    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with (output / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_manifest(output: Path, rows: list[dict[str, str]]) -> None:
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "source_image",
                "source_label",
                "output_image",
                "output_label",
                "label_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    output.mkdir(parents=True, exist_ok=True)

    labels_by_stem = index_labels(source)
    samples = [Sample(image=image, label=choose_label(image, labels_by_stem)) for image in iter_images(source)]

    if not samples:
        raise SystemExit(f"No images found under {source}")

    if args.require_labels:
        missing = [sample.image for sample in samples if sample.label is None]
        if missing:
            raise SystemExit(f"{len(missing)} images have no label file.")

    splits = split_samples(samples, args.val_ratio, args.test_ratio, args.seed)
    rows: list[dict[str, str]] = []
    for split, split_samples_ in splits.items():
        rows.extend(copy_split(split, split_samples_, source, output))

    write_dataset_yaml(output, load_names(args.names))
    write_manifest(output, rows)

    print(f"Wrote {len(samples)} samples to {output}")
    for split, split_samples_ in splits.items():
        print(f"{split}: {len(split_samples_)}")


if __name__ == "__main__":
    main()
