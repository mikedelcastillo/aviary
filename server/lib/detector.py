"""YOLO object detector wrapper and image annotation helpers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2

from lib.config import ModelConfig
from lib.labels import format_confidence, pretty


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(self, config: ModelConfig) -> None:
        for path in config.paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file does not exist: {path}")

        from ultralytics import YOLO

        self.config = config
        # Each model runs its own pass per frame; detections are concatenated.
        self.models = [YOLO(str(path)) for path in config.paths]
        self._lock = threading.Lock()

    def known_labels(self) -> list[str]:
        """Every class label across all loaded models, sorted and de-duplicated.

        This is the authoritative runtime set of things the server can detect —
        the roster's individual birds (percy, matcha, ...), the IR species
        labels, and any generic classes from a stock YOLO model. ``/find`` uses
        it to validate a requested target and to list what is findable.
        """
        labels: set[str] = set()
        for model in self.models:
            names = model.names
            values = names.values() if isinstance(names, dict) else names
            labels.update(str(name) for name in values)
        return sorted(labels)

    def predict(self, frame) -> list[Detection]:
        predict_args = {
            "source": frame,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.image_size,
            "verbose": False,
        }
        if self.config.device != "auto":
            predict_args["device"] = self.config.device

        detections: list[Detection] = []
        # One lock still serializes all inference so the models never contend
        # for the GPU; each model adds its detections to the merged list.
        with self._lock:
            for model in self.models:
                results = model.predict(**predict_args)
                if not results:
                    continue

                names = model.names
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    raw_xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = (int(round(value)) for value in raw_xyxy)

                    if isinstance(names, dict):
                        label = str(names.get(cls_id, cls_id))
                    else:
                        label = str(names[cls_id]) if cls_id < len(names) else str(cls_id)

                    detections.append(
                        Detection(
                            label=label,
                            confidence=confidence,
                            bbox_xyxy=(x1, y1, x2, y2),
                        )
                    )

        return detections


def draw_detections(frame, detections: list[Detection]):
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        text = f"{pretty(detection.label)} {format_confidence(detection.confidence)}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated
