"""Raw frame collection for alerted detections."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from lib.config import CameraConfig, CollectConfig
from lib.detector import Detection


def safe_path_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned or "unknown"


def collection_stem(camera_name: str, label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{safe_path_component(camera_name)}_{safe_path_component(label)}_{uuid4().hex[:8]}"


def should_collect(config: CollectConfig, detection: Detection) -> bool:
    return detection.label.strip().lower() in config.objects


def write_collection_instance(
    collect_dir: Path,
    camera: CameraConfig,
    frame,
    detection: Detection,
) -> tuple[Path, Path]:
    object_dir = collect_dir / safe_path_component(detection.label)
    object_dir.mkdir(parents=True, exist_ok=True)

    stem = collection_stem(camera.name, detection.label)
    image_path = object_dir / f"{stem}.jpg"
    metadata_path = object_dir / f"{stem}.json"

    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write collected image: {image_path}")

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    metadata = {
        "camera": asdict(camera),
        "object": detection.label,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "image_file": image_path.name,
        "frame": {
            "width": width,
            "height": height,
        },
        "detection": {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox_xyxy": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "bbox_width": x2 - x1,
            "bbox_height": y2 - y1,
            "center": {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
            },
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path, metadata_path


def collect_alerted_detections(
    config: CollectConfig,
    camera: CameraConfig,
    frame,
    detections: list[Detection],
) -> list[tuple[Path, Path]]:
    if not config.objects:
        return []

    collected = []
    for detection in detections:
        if should_collect(config, detection):
            collected.append(write_collection_instance(config.directory, camera, frame, detection))
    return collected
