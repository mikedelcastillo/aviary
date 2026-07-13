"""Privacy screen: keep photos with people in them off Telegram.

The cameras watch the birds, but the owners walk through the frame too. Before
any image is uploaded to Telegram, the notifier runs it through this screen —
a stock COCO YOLO pass that answers "is a person visible?". A flagged photo is
withheld; the alert/report text still goes out with a note explaining why the
photo is absent. Only the Telegram upload path is screened: local snapshots,
collection and memory photos never leave the machine and are untouched.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from lib.config import PrivacyConfig

LOGGER = logging.getLogger("lib.privacy")

PERSON_LABEL = "person"


class PersonScreen:
    """Answers "is a person in this JPEG?" with a small stock-COCO model.

    Deliberately recall-biased and FAIL-CLOSED: a missed person leaks a face to
    Telegram, while a false positive merely withholds one bird photo — so the
    confidence floor is low, and an image that can't be decoded or inferred on
    counts as containing a person.
    """

    def __init__(self, config: PrivacyConfig) -> None:
        # Fail fast on a missing EXPLICIT path (a typo'd PRIVACY_MODEL_PATH
        # should stop the boot, not silently withhold every photo). A bare
        # stock name like the default "yolo11n.pt" is exempt: *.pt files are
        # gitignored, and ultralytics auto-downloads official weights by name,
        # so a fresh checkout boots without a manual model fetch.
        path = Path(config.model_path)
        if not path.exists() and path.name != str(path):
            raise FileNotFoundError(
                f"Privacy model file does not exist: {config.model_path}"
            )

        from ultralytics import YOLO

        self.config = config
        self._model = YOLO(str(config.model_path))
        # YOLO predict is not thread-safe; sends arrive from several threads
        # (alert worker, caretaker beat, find/command threads).
        self._lock = threading.Lock()

    def has_person(self, image_bytes: bytes) -> bool:
        array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if array is None:
            LOGGER.warning("Privacy screen could not decode image; withholding it")
            return True

        try:
            with self._lock:
                results = self._model.predict(
                    source=array,
                    conf=self.config.confidence,
                    imgsz=self.config.image_size,
                    device=self.config.device,
                    verbose=False,
                )
        except Exception:
            LOGGER.exception("Privacy screen inference failed; withholding image")
            return True

        if not results:
            return False
        names = self._model.names
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if isinstance(names, dict):
                label = names.get(cls_id, "")
            else:
                label = names[cls_id] if cls_id < len(names) else ""
            # Match by class NAME, not COCO index 0, so any person-aware model
            # works as the screen — not just stock COCO orderings.
            if str(label).lower() == PERSON_LABEL:
                LOGGER.info(
                    "Privacy screen: person in frame (conf=%.2f); photo withheld",
                    float(box.conf[0].item()),
                )
                return True
        return False
