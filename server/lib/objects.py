"""Object sighting registry shared across cameras."""

from __future__ import annotations

import threading
import time

from lib.detector import Detection


FrameSize = tuple[int, int]
Point = tuple[float, float]


def detection_center(detection: Detection) -> Point:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def frame_size_from_shape(frame_shape) -> FrameSize:
    height, width = frame_shape[:2]
    return (int(width), int(height))


def movement_ratio(previous_center: Point, center: Point, frame_size: FrameSize) -> float:
    width, height = frame_size
    x_ratio = abs(center[0] - previous_center[0]) / width if width > 0 else 0.0
    y_ratio = abs(center[1] - previous_center[1]) / height if height > 0 else 0.0
    return max(x_ratio, y_ratio)


class ObjectRegistry:
    """Camera-specific tally of every label seen."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[tuple[str, str], dict] = {}

    def record(self, detections: list[Detection], camera_name: str, frame_size: FrameSize) -> None:
        now = time.monotonic()
        recorded_labels: set[str] = set()
        with self._lock:
            for detection in detections:
                if detection.label in recorded_labels:
                    continue
                recorded_labels.add(detection.label)

                key = (camera_name, detection.label)
                center = detection_center(detection)
                entry = self._objects.get(key)
                if entry is None:
                    entry = {
                        "count": 0,
                        "center": None,
                        "frame_size": frame_size,
                        "last_alert": None,
                    }
                    self._objects[key] = entry

                previous_center = entry["center"]
                movement = (
                    None
                    if previous_center is None
                    else movement_ratio(previous_center, center, frame_size)
                )
                entry["last_seen"] = now
                entry["count"] += 1
                entry["previous_center"] = previous_center
                entry["center"] = center
                entry["frame_size"] = frame_size
                entry["movement_ratio"] = movement

    def record_alert(self, detections: list[Detection], camera_name: str) -> None:
        now = time.monotonic()
        recorded_labels: set[str] = set()
        with self._lock:
            for detection in detections:
                if detection.label in recorded_labels:
                    continue
                recorded_labels.add(detection.label)

                key = (camera_name, detection.label)
                entry = self._objects.get(key)
                if entry is not None:
                    entry["last_alert"] = now

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            rows = [
                {
                    "camera": camera_name,
                    "label": label,
                    "since": now - entry["last_seen"],
                    "count": entry["count"],
                    "center": entry["center"],
                    "previous_center": entry.get("previous_center"),
                    "frame_size": entry["frame_size"],
                    "movement_ratio": entry.get("movement_ratio"),
                    "movement_percent": (
                        None
                        if entry.get("movement_ratio") is None
                        else entry["movement_ratio"] * 100
                    ),
                    "since_alert": (
                        None if entry.get("last_alert") is None else now - entry["last_alert"]
                    ),
                }
                for (camera_name, label), entry in self._objects.items()
            ]
        # Smallest "since" (most recently seen) first.
        rows.sort(key=lambda row: row["since"])
        return rows
