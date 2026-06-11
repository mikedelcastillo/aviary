"""Alert eligibility, snapshot writing, and asynchronous delivery."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from lib.collect import collect_alerted_detections
from lib.config import AppConfig, CameraConfig
from lib.detector import Detection, draw_detections
from lib.objects import FrameSize, detection_center, movement_ratio
from lib.telegram.notifier import TelegramNotifier


LOGGER = logging.getLogger("lib.alerts")


def snapshot_name(camera_name: str) -> str:
    safe_camera = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in camera_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{safe_camera}.jpg"


def write_snapshot(snapshot_dir: Path, camera_name: str, frame, detections: list[Detection]) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / snapshot_name(camera_name)
    annotated = draw_detections(frame, detections)
    ok = cv2.imwrite(str(path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to write snapshot: {path}")
    return path


class AlertState:
    def __init__(
        self,
        last_seen_alert_seconds: float,
        bbox_movement_alert_ratio: float,
        filter_objects: frozenset[str] = frozenset(),
    ) -> None:
        self.last_seen_alert_seconds = last_seen_alert_seconds
        self.bbox_movement_alert_ratio = bbox_movement_alert_ratio
        # When non-empty, only these labels are allowed to alert. Empty means no
        # filtering. Mirrors COLLECT_OBJECTS membership semantics (lowercased).
        self.filter_objects = filter_objects
        self._objects: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def _passes_filter(self, detection: Detection) -> bool:
        if not self.filter_objects:
            return True
        return detection.label.strip().lower() in self.filter_objects

    def eligible(
        self,
        camera_name: str,
        detections: list[Detection],
        frame_size: FrameSize,
    ) -> list[Detection]:
        now = time.monotonic()
        selected: list[Detection] = []
        seen_labels: set[str] = set()

        with self._lock:
            for detection in detections:
                if not self._passes_filter(detection):
                    continue
                if detection.label in seen_labels:
                    continue
                seen_labels.add(detection.label)

                key = (camera_name, detection.label)
                center = detection_center(detection)
                entry = self._objects.get(key)

                should_alert = False
                if entry is None:
                    should_alert = True
                elif now - entry["last_alert_at"] > self.last_seen_alert_seconds:
                    should_alert = True
                else:
                    moved = movement_ratio(entry["center"], center, frame_size)
                    should_alert = moved >= self.bbox_movement_alert_ratio

                if should_alert:
                    selected.append(detection)

                updated = {
                    "last_alert_at": now if should_alert else entry["last_alert_at"],
                    "last_seen_at": now,
                    "center": center,
                    "frame_size": frame_size,
                }
                self._objects[key] = updated

        return selected


@dataclass
class AlertJob:
    camera: CameraConfig
    frame: object
    detections: list[Detection]


# Cap on undelivered alerts held in memory. Alert eligibility already limits
# repeated enqueues, so this only bites during a sustained Telegram outage.
# Dropping backlog is preferable to growing memory without bound.
ALERT_QUEUE_MAXSIZE = 32


class AlertDispatcher:
    """Delivers alerts (snapshot + Telegram) off the camera capture threads."""

    def __init__(
        self,
        app_config: AppConfig,
        notifier: TelegramNotifier | None,
        stop_event: threading.Event,
        workers: int,
    ) -> None:
        self._app_config = app_config
        self._notifier = notifier
        self._stop_event = stop_event
        self._queue: "queue.Queue[AlertJob]" = queue.Queue(maxsize=ALERT_QUEUE_MAXSIZE)
        self._workers = [
            threading.Thread(target=self._run, name=f"alert-dispatch-{index}", daemon=True)
            for index in range(max(1, workers))
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, camera: CameraConfig, frame, detections: list[Detection]) -> None:
        # Copy the frame: delivery happens later on a worker thread, decoupled
        # from this capture loop's frame lifetime.
        job = AlertJob(camera=camera, frame=frame.copy(), detections=detections)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            LOGGER.warning(
                "Alert queue full (%d); dropping alert for camera=%s - delivery is backed up",
                ALERT_QUEUE_MAXSIZE,
                camera.name,
            )

    def _run(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            try:
                self._deliver(job)
            except Exception:
                LOGGER.exception("Alert delivery failed for camera=%s", job.camera.name)
            finally:
                self._queue.task_done()

    def _deliver(self, job: AlertJob) -> None:
        collected = []
        try:
            collected = collect_alerted_detections(
                self._app_config.collect, job.camera, job.frame, job.detections
            )
        except Exception:
            LOGGER.exception("Collection failed for camera=%s", job.camera.name)

        snapshot_path = None
        if self._app_config.telegram.include_snapshot:
            snapshot_path = write_snapshot(
                self._app_config.snapshot_dir, job.camera.name, job.frame, job.detections
            )

        labels = ", ".join(sorted({detection.label for detection in job.detections}))
        LOGGER.info(
            "Alerting camera=%s labels=%s snapshot=%s collected=%d",
            job.camera.name,
            labels,
            snapshot_path,
            len(collected),
        )

        try:
            if self._notifier:
                self._notifier.send_detections(job.camera.name, job.detections, snapshot_path)
        finally:
            if snapshot_path is not None:
                snapshot_path.unlink(missing_ok=True)

    def shutdown(self, timeout: float = 5.0) -> None:
        # stop_event is already set by the caller; workers drain any in-flight
        # jobs then exit on the next empty poll. Join so a final alert isn't cut
        # off mid-upload during a clean shutdown.
        for worker in self._workers:
            worker.join(timeout=timeout)
