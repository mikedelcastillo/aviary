"""Pretrained bird detector used as an Immich prefilter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BirdPrediction:
    has_bird: bool
    max_confidence: float
    detections: list[dict[str, float | int | str]]


def select_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return "mps"

    return "cpu"


class PretrainedBirdDetector:
    def __init__(
        self,
        model_name: str,
        threshold: float,
        device: str = "auto",
        bird_labels: Iterable[str] = ("bird",),
    ) -> None:
        from ultralytics import YOLO

        self.model_name = model_name
        self.threshold = threshold
        self.device = select_device(device)
        # Half precision (fp16) only applies on CUDA GPUs; cpu/mps run in fp32.
        self.half = self.device.startswith("cuda")
        self.model = YOLO(model_name)
        self.bird_labels = {label.lower() for label in bird_labels}
        self.bird_class_ids = self._resolve_bird_class_ids()
        if not self.bird_class_ids:
            raise ValueError(f"Model {model_name} does not expose any of these labels: {sorted(self.bird_labels)}")
        self._warmup()

    def _warmup(self) -> None:
        """Run a throwaway inference so the first real batch doesn't eat compile cost."""
        try:
            import numpy as np

            blank = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(
                source=blank,
                conf=self.threshold,
                device=self.device,
                classes=sorted(self.bird_class_ids),
                half=self.half,
                verbose=False,
            )
        except Exception:
            pass

    def predict(self, image_path: Path) -> BirdPrediction:
        return self.predict_batch([image_path], batch_size=1)[0]

    def predict_batch(self, image_paths: Iterable[Path], batch_size: int = 64) -> list[BirdPrediction]:
        paths = list(image_paths)
        if not paths:
            return []

        results = self.model.predict(
            source=[str(path) for path in paths],
            batch=max(1, batch_size),
            conf=self.threshold,
            device=self.device,
            classes=sorted(self.bird_class_ids),
            half=self.half,
            verbose=False,
        )

        return [self._prediction_from_result(result) for result in results]

    def _prediction_from_result(self, result) -> BirdPrediction:
        detections: list[dict[str, float | int | str]] = []
        max_confidence = 0.0

        names = self.model.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            label = str(names[class_id] if not isinstance(names, dict) else names.get(class_id, class_id))
            max_confidence = max(max_confidence, confidence)
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return BirdPrediction(
            has_bird=bool(detections),
            max_confidence=max_confidence,
            detections=detections,
        )

    def _resolve_bird_class_ids(self) -> set[int]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(class_id) for class_id, label in names.items() if str(label).lower() in self.bird_labels}
        return {index for index, label in enumerate(names) if str(label).lower() in self.bird_labels}
