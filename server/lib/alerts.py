"""Alert eligibility, snapshot writing, and asynchronous delivery."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from lib.collect import collect_alerted_detections
from lib.config import AppConfig, CameraConfig
from lib.detector import Detection, draw_detections
from lib.objects import FrameSize, detection_center, movement_ratio
from lib.telegram.notifier import TelegramNotifier


LOGGER = logging.getLogger("lib.alerts")

# Cap on alert anchors retained per (camera, label). Anchors expire on the
# last-alert window anyway; this only bounds the per-frame movement scan when an
# object relentlessly relocates to brand-new positions.
ANCHOR_LIMIT = 50


def snapshot_name(camera_name: str) -> str:
    safe_camera = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in camera_name)
    # UTC like the collect/snapshot sidecar timestamps, so filename ordering is
    # monotonic and consistent across the trees (a DST host won't reorder them).
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
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
        # Per (camera, normalized-label) alert state. Each entry holds a list of
        # "anchors" — positions we have already alerted on, each with the time of
        # that alert: ``{"anchors": [{"center": (x, y), "last_alert_at": t}, ...]}``.
        # Anchoring on alerted positions (not the last-seen position) is what makes
        # throttling correct across multiple models: extra or jittering boxes for
        # the same object land near an existing anchor and are suppressed, so the
        # last-alert and movement rules hold no matter how many models contribute.
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

        # Group this frame's alertable detections by normalized label so that
        # several models (or duplicate boxes from one model) describing the same
        # object collapse into a single per-label decision. Case/whitespace are
        # folded so two models labelling "Bird"/"bird" share one alert stream.
        by_label: dict[str, list[Detection]] = {}
        for detection in detections:
            if not self._passes_filter(detection):
                continue
            by_label.setdefault(detection.label.strip().lower(), []).append(detection)

        with self._lock:
            for label_key, group in by_label.items():
                representative = self._eligible_for_label(
                    (camera_name, label_key), group, frame_size, now
                )
                if representative is not None:
                    selected.append(representative)

        return selected

    def _eligible_for_label(
        self,
        key: tuple[str, str],
        group: list[Detection],
        frame_size: FrameSize,
        now: float,
    ) -> Detection | None:
        """Decide whether ``key``'s label should alert this frame.

        Returns the representative detection to deliver (the strongest box), or
        ``None`` when no alert is due. Caller holds ``self._lock``.
        """
        entry = self._objects.get(key)
        # Keep only anchors still inside the last-alert window. An expired anchor
        # means it has been at least ``last_seen_alert_seconds`` since we alerted
        # for that spot, so a still-present object there earns a re-confirmation.
        anchors: list[dict] = []
        if entry is not None:
            anchors = [
                anchor
                for anchor in entry["anchors"]
                if now - anchor["last_alert_at"] < self.last_seen_alert_seconds
            ]

        representative: Detection | None = None
        # Strongest detections first so the alert (and its anchor) use the most
        # confident box when several describe the same object.
        for detection in sorted(group, key=lambda d: d.confidence, reverse=True):
            center = detection_center(detection)
            near_existing = any(
                movement_ratio(anchor["center"], center, frame_size)
                < self.bbox_movement_alert_ratio
                for anchor in anchors
            )
            if near_existing:
                # Close to a position we have already alerted on within the
                # window: same sighting, model jitter, or a redundant box — skip.
                continue
            # A genuinely new position: first sighting, a relocation past the
            # movement threshold, or the first frame after the window lapsed.
            anchors.append({"center": center, "last_alert_at": now})
            if representative is None:
                representative = detection

        if len(anchors) > ANCHOR_LIMIT:
            anchors = anchors[-ANCHOR_LIMIT:]
        self._objects[key] = {"anchors": anchors}
        return representative


@dataclass
class AlertJob:
    camera: CameraConfig
    frame: object
    detections: list[Detection]


@dataclass
class TelegramJob:
    """A prepared alert awaiting Telegram delivery (the only stage that pauses)."""

    camera: CameraConfig
    detections: list[Detection]
    snapshot_path: Path | None


# Cap on undelivered alerts held in memory. Alert eligibility already limits
# repeated enqueues, so this only bites during a sustained Telegram outage.
# Dropping backlog is preferable to growing memory without bound.
ALERT_QUEUE_MAXSIZE = 32
# Pending Telegram sends. This backs up on its own when Telegram rate-limits us
# (429); when full we drop the OLDEST pending message so the freshest alerts win
# and collection — which already happened upstream — is never affected.
TELEGRAM_QUEUE_MAXSIZE = 32


class AlertDispatcher:
    """Delivers alerts off the camera capture threads.

    Two stages, deliberately decoupled. The *prepare* workers run collection and
    snapshot writing; the single *Telegram* worker performs the actual API send.
    Telegram is the only thing that can pause (rate-limit backoff), so isolating
    it on its own queue/worker means a 429 never stalls or drops collection: the
    prepare workers keep draining frames while the Telegram backlog waits.
    """

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

        # Telegram delivery is serialized on a single worker behind its own
        # queue, so its rate-limit pauses are contained here and never reach the
        # prepare workers above. Only spun up when there is a notifier to feed.
        self._telegram_queue: "queue.Queue[TelegramJob]" = queue.Queue(
            maxsize=TELEGRAM_QUEUE_MAXSIZE
        )
        self._telegram_enqueue_lock = threading.Lock()
        self._telegram_worker: threading.Thread | None = None
        if self._notifier is not None:
            self._telegram_worker = threading.Thread(
                target=self._run_telegram, name="alert-telegram", daemon=True
            )
            self._telegram_worker.start()

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
                self._prepare(job)
            except Exception:
                LOGGER.exception("Alert preparation failed for camera=%s", job.camera.name)
            finally:
                self._queue.task_done()

    def _prepare(self, job: AlertJob) -> None:
        """Collect and snapshot the alert, then hand the send to the Telegram worker.

        Runs on the prepare workers. Everything here is local work that must
        always happen, so it is kept off the Telegram path that can pause.
        """
        # Collection is independent of Telegram and must always run, even while
        # Telegram delivery is paused by a 429 backoff.
        collected = []
        try:
            collected = collect_alerted_detections(
                self._app_config.collect, job.camera, job.frame, job.detections
            )
        except Exception:
            LOGGER.exception("Collection failed for camera=%s", job.camera.name)

        # Raw per-detection photo alerts can be switched off (e.g. when the
        # daycare digest replaces them); collection above still always runs.
        send_alert = self._notifier is not None and self._app_config.raw_photo_alerts
        snapshot_path = None
        if send_alert and self._app_config.telegram.include_snapshot:
            try:
                snapshot_path = write_snapshot(
                    self._app_config.snapshot_dir, job.camera.name, job.frame, job.detections
                )
            except Exception:
                LOGGER.exception("Snapshot write failed for camera=%s", job.camera.name)

        labels = ", ".join(sorted({detection.label for detection in job.detections}))
        LOGGER.info(
            "Alerting camera=%s labels=%s snapshot=%s collected=%d send_alert=%s",
            job.camera.name,
            labels,
            snapshot_path,
            len(collected),
            send_alert,
        )

        if not send_alert:
            # No raw alert to deliver (digest mode or no Telegram); collection done.
            if snapshot_path is not None:
                snapshot_path.unlink(missing_ok=True)
            return

        self._enqueue_telegram(
            TelegramJob(
                camera=job.camera,
                detections=job.detections,
                snapshot_path=snapshot_path,
            )
        )

    def _enqueue_telegram(self, telegram_job: TelegramJob) -> None:
        # Drop the OLDEST pending send when the Telegram backlog is full (e.g. a
        # sustained 429 pause) so the freshest alerts survive. The lock keeps the
        # drop-then-put atomic across concurrent prepare workers.
        with self._telegram_enqueue_lock:
            while True:
                try:
                    self._telegram_queue.put_nowait(telegram_job)
                    return
                except queue.Full:
                    try:
                        dropped = self._telegram_queue.get_nowait()
                    except queue.Empty:
                        continue
                    self._telegram_queue.task_done()
                    if dropped.snapshot_path is not None:
                        dropped.snapshot_path.unlink(missing_ok=True)
                    LOGGER.warning(
                        "Telegram backlog full (%d); dropped oldest alert for camera=%s",
                        TELEGRAM_QUEUE_MAXSIZE,
                        dropped.camera.name,
                    )

    def _run_telegram(self) -> None:
        while True:
            try:
                telegram_job = self._telegram_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            try:
                if self._notifier is not None:
                    self._notifier.send_detections(
                        telegram_job.camera.name,
                        telegram_job.detections,
                        telegram_job.snapshot_path,
                    )
            except Exception:
                LOGGER.exception(
                    "Telegram send failed for camera=%s", telegram_job.camera.name
                )
            finally:
                if telegram_job.snapshot_path is not None:
                    telegram_job.snapshot_path.unlink(missing_ok=True)
                self._telegram_queue.task_done()

    def shutdown(self, timeout: float = 5.0) -> None:
        # stop_event is already set by the caller; workers drain any in-flight
        # jobs then exit on the next empty poll. Join so a final alert isn't cut
        # off mid-upload during a clean shutdown. The Telegram worker may be in a
        # rate-limit pause, so its join is best-effort within the timeout.
        for worker in self._workers:
            worker.join(timeout=timeout)
        if self._telegram_worker is not None:
            self._telegram_worker.join(timeout=timeout)
