"""YOLO object detector wrapper and image annotation helpers."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from lib.config import ModelConfig
from lib.labels import format_confidence, pretty

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


def pick_freest_device(free_bytes_by_index: list[int]) -> str | None:
    """The ``cuda:N`` with the most free VRAM, or None when the default is fine.

    On a single-GPU (or no-GPU) box ultralytics' own default is already right;
    only a multi-GPU box needs steering, so detection lands on the idle card
    instead of piling onto cuda:0 with the Ollama models.
    """
    if len(free_bytes_by_index) < 2:
        return None
    return f"cuda:{max(range(len(free_bytes_by_index)), key=free_bytes_by_index.__getitem__)}"


def _resolve_auto_device() -> str | None:
    """Probe CUDA for the freest GPU; None leaves the choice to ultralytics."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    except Exception:  # CUDA probing must never break detector startup
        return None
    return pick_freest_device(free)


class ObjectDetector:
    def __init__(self, config: ModelConfig) -> None:
        for path in config.paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file does not exist: {path}")

        from ultralytics import YOLO

        self.config = config
        # "auto" on a multi-GPU box: pin to the GPU with the most free VRAM,
        # measured once at startup. ponytail: no mid-run rebalancing — the pick
        # only goes stale if Ollama's placement changes under a running server,
        # and a restart re-picks.
        self.device = config.device if config.device != "auto" else _resolve_auto_device()
        if self.device:
            LOGGER.info("Detector device: %s", self.device)
        # Each model runs its own pass per frame; detections are concatenated.
        self.models = [YOLO(str(path)) for path in config.paths]
        self._lock = threading.Lock()

    def warmup(self) -> None:
        """Run one throwaway predict per model so the first real frame is fast.

        The first CUDA inference pays context init + cuDNN autotune (seconds);
        paying it at boot keeps the first camera frame's latency flat. Separate
        from ``__init__`` so construction stays side-effect-free for tests.
        """
        import numpy as np

        dummy = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self.predict(dummy)
        except Exception:  # warmup must never block startup
            LOGGER.exception("Detector warmup failed")
            return
        LOGGER.info("Detector warmed up in %.1fs", time.perf_counter() - started)

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
            # Pascal (sm_61) runs fp16 at ~1/64 fp32 throughput — pin full
            # precision explicitly rather than trusting the upstream default.
            "half": False,
        }
        if self.device:
            predict_args["device"] = self.device

        detections: list[Detection] = []
        # One lock still serializes all inference so the models never contend
        # for the GPU; each model adds its detections to the merged list.
        with self._lock:
            for model in self.models:
                results = self._predict_resilient(model, predict_args)
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

    def _predict_resilient(self, model, predict_args: dict):
        """One model predict; on CUDA OOM, re-pick the freest GPU and retry once.

        The detector shares its GPU with Ollama, whose residency changes under
        a running server (models load/unload on demand). A single OOM therefore
        means "my startup pick went stale", not "give up": free the cache,
        re-probe, and retry on whatever card now has the most room. A second
        failure propagates — the camera loop already logs and survives it.
        """
        try:
            return model.predict(**predict_args)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            LOGGER.warning("Detector CUDA OOM on %s; re-picking device", self.device)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            repicked = _resolve_auto_device()
            if repicked:
                self.device = repicked
                predict_args = {**predict_args, "device": repicked}
                LOGGER.info("Detector device re-picked: %s", repicked)
            return model.predict(**predict_args)


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
