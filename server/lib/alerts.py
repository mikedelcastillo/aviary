"""Alert throttling, snapshot writing, and asynchronous delivery."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from lib.config import AppConfig, CameraConfig
from lib.detector import Detection, draw_detections
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
    def __init__(self, cooldown_seconds: int) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_sent: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def eligible(self, camera_name: str, detections: list[Detection]) -> list[Detection]:
        now = time.monotonic()
        selected: list[Detection] = []
        selected_labels: set[str] = set()

        with self._lock:
            for detection in detections:
                key = (camera_name, detection.label)
                if detection.label in selected_labels:
                    continue

                last_sent = self._last_sent.get(key, 0)
                if now - last_sent >= self.cooldown_seconds:
                    selected.append(detection)
                    selected_labels.add(detection.label)

            for detection in selected:
                self._last_sent[(camera_name, detection.label)] = now

        return selected


@dataclass
class AlertJob:
    camera: CameraConfig
    frame: object
    detections: list[Detection]


# Cap on undelivered alerts held in memory. The cooldown already throttles
# enqueues to roughly one per camera per cooldown window, so this only bites
# during a sustained Telegram outage. Dropping backlog is preferable to growing
# memory without bound.
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
        snapshot_path = None
        if self._app_config.telegram.include_snapshot:
            snapshot_path = write_snapshot(
                self._app_config.snapshot_dir, job.camera.name, job.frame, job.detections
            )

        labels = ", ".join(sorted({detection.label for detection in job.detections}))
        LOGGER.info("Alerting camera=%s labels=%s snapshot=%s", job.camera.name, labels, snapshot_path)

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
