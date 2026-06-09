"""YOLO object-detection backend for the model abstraction.

Thin :class:`~aviary_immich.models.base.Model` adapter over the existing
:class:`aviary_immich.detector.PretrainedBirdDetector`: it forwards both the array and path entry
points unchanged (so the adaptive OOM-halving in :mod:`aviary_immich.inference` keeps working) and
maps each ``BirdPrediction`` into a :class:`~aviary_immich.models.base.ModelOutput` of
lowercased-name :class:`~aviary_immich.models.base.Tag`\\ s.

The detector pulls in torch/cv2/ultralytics, so it is imported lazily inside ``__init__`` — importing
this module by itself stays cheap.
"""

from __future__ import annotations

from typing import Any

from aviary_immich.models.base import ModelOutput, Tag


class YoloObjectModel:
    """A YOLO detector exposed through the :class:`~aviary_immich.models.base.Model` protocol."""

    def __init__(self, model_name: str, threshold: float, device: str, labels: tuple[str, ...]):
        from aviary_immich.detector import PretrainedBirdDetector

        self._detector = PretrainedBirdDetector(model_name, threshold, device, labels)
        self.name = "yolo"
        self.half = self._detector.half
        # The OOM-halving code in inference.py reads/writes this on the model.
        self._infer_cap = None

    def predict_arrays(self, images: list[Any], batch_size: int) -> list[ModelOutput]:
        return [self._to_output(p) for p in self._detector.predict_batch_arrays(images, batch_size)]

    def predict_paths(self, images: list[Any], batch_size: int) -> list[ModelOutput]:
        return [self._to_output(p) for p in self._detector.predict_batch(images, batch_size)]

    @staticmethod
    def _to_output(prediction) -> ModelOutput:
        tags = tuple(
            Tag(
                str(detection["label"]).lower(),
                float(detection["confidence"]),
                (
                    int(detection["x1"]),
                    int(detection["y1"]),
                    int(detection["x2"]),
                    int(detection["y2"]),
                ),
            )
            for detection in prediction.detections
        )
        return ModelOutput(tags=tags, max_confidence=float(prediction.max_confidence))
