"""YOLO detector wrapper and image annotation helpers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2

from aviary_server.config import ModelConfig


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    zone: str | None = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class BirdDetector:
    def __init__(self, config: ModelConfig) -> None:
        if not Path(config.path).exists():
            raise FileNotFoundError(f"Model file does not exist: {config.path}")

        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(str(config.path))
        self._lock = threading.Lock()

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

        with self._lock:
            results = self.model.predict(**predict_args)

        if not results:
            return []

        names = self.model.names
        detections: list[Detection] = []
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
        text = f"{detection.label} {detection.confidence:.2f}"
        if detection.zone:
            text = f"{text} [{detection.zone}]"

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
