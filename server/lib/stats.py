"""Thread-safe runtime statistics for camera workers."""

from __future__ import annotations

import threading
import time

from lib.detector import Detection
from lib.objects import FrameSize, ObjectRegistry


class CameraStats:
    """Thread-safe rolling metrics for a single camera.

    The owning capture thread calls the ``record_*``/``set_status`` mutators;
    the dashboard thread calls :meth:`snapshot`. All access is guarded so a
    repaint never sees a half-updated set of counters.
    """

    def __init__(
        self,
        name: str,
        sample_fps: float,
        registry: ObjectRegistry | None = None,
        fps_window: float = 2.0,
        filter_objects: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.sample_fps = sample_fps
        self._registry = registry
        self._fps_window = fps_window
        # When non-empty, only these labels are surfaced in the terminal stats
        # (the "detect" row and the objects-by-camera panel). Everything else is
        # still detected, but kept out of the display. Mirrors FILTER_OBJECTS /
        # AlertState membership semantics (lowercased).
        self._filter_objects = filter_objects
        self._lock = threading.Lock()

        # Most-recent decoded frame, published by the capture loop for on-demand
        # snapshots (the /snapshot command). Guarded by its OWN lock so grabbing a
        # frame never contends with the dashboard's counter reads, and the store
        # is a cheap reference swap on the hot path (the copy is paid by the rare
        # reader instead). ``cv2.VideoCapture.read`` hands back a fresh array each
        # call, so the stored reference is never mutated in place after this.
        self._latest_frame = None
        self._latest_frame_at: float | None = None
        self._frame_lock = threading.Lock()

        self.status = "connecting"
        self.backoff = 0.0
        self.frames_total = 0
        self.detections_total = 0
        self.alerts_sent = 0
        self.reconnects = 0
        self.consecutive_failures = 0
        self.fps = 0.0
        self.last_label: str | None = None
        self.last_detection_at: float | None = None
        self.last_frame_at: float | None = None
        self.started_at = time.monotonic()

        self._fps_count = 0
        self._fps_window_start = time.monotonic()

    def set_status(self, status: str, backoff: float = 0.0) -> None:
        with self._lock:
            self.status = status
            self.backoff = backoff

    def record_inference(
        self,
        labels: list[str],
        detections: list[Detection] | None = None,
        frame_size: FrameSize | None = None,
    ) -> None:
        now = time.monotonic()
        if self._filter_objects:
            labels = [label for label in labels if label.strip().lower() in self._filter_objects]
            if detections is not None:
                detections = [
                    detection
                    for detection in detections
                    if detection.label.strip().lower() in self._filter_objects
                ]
        with self._lock:
            self.frames_total += 1
            self.last_frame_at = now
            self.consecutive_failures = 0
            self._fps_count += 1
            elapsed = now - self._fps_window_start
            if elapsed >= self._fps_window:
                self.fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_window_start = now
            if labels:
                self.detections_total += len(labels)
                self.last_label = ", ".join(sorted(set(labels)))
                self.last_detection_at = now
        # Update the shared registry outside this camera's lock to avoid holding
        # two locks at once.
        if detections and frame_size is not None and self._registry is not None:
            self._registry.record(detections, self.name, frame_size)

    def set_latest_frame(self, frame) -> None:
        """Publish the newest decoded frame for on-demand snapshots.

        Called from the capture loop on every successful read. Stores the
        reference only (no copy) so the hot path stays cheap; readers copy.
        """
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_at = time.monotonic()

    def latest_frame(self) -> tuple[object | None, float | None]:
        """Return ``(frame_copy, age_seconds)`` or ``(None, None)`` if unseen.

        The frame is copied under the lock so the caller owns it outright and the
        capture loop can keep replacing the stored reference. ``age_seconds`` is
        how long ago that frame was captured, so a snapshot can flag a stalled or
        disconnected camera whose last good frame is now stale.
        """
        with self._frame_lock:
            if self._latest_frame is None or self._latest_frame_at is None:
                return None, None
            return self._latest_frame.copy(), time.monotonic() - self._latest_frame_at

    def record_read_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1

    def record_reconnect(self) -> None:
        # A torn-down session: zero the live rate so a stalled cell doesn't keep
        # showing a stale FPS from before the drop.
        with self._lock:
            self.reconnects += 1
            self.fps = 0.0
            self._fps_count = 0
            self._fps_window_start = time.monotonic()

    def record_alert(self, count: int = 1) -> None:
        with self._lock:
            self.alerts_sent += count

    def record_object_alert(self, detections: list[Detection]) -> None:
        if self._registry is not None:
            self._registry.record_alert(detections, self.name)

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            since_frame = None if self.last_frame_at is None else now - self.last_frame_at
            since_detection = (
                None if self.last_detection_at is None else now - self.last_detection_at
            )
            return {
                "name": self.name,
                "sample_fps": self.sample_fps,
                "status": self.status,
                "backoff": self.backoff,
                "frames_total": self.frames_total,
                "detections_total": self.detections_total,
                "alerts_sent": self.alerts_sent,
                "reconnects": self.reconnects,
                "consecutive_failures": self.consecutive_failures,
                "fps": self.fps,
                "last_label": self.last_label,
                "since_detection": since_detection,
                "since_frame": since_frame,
                "uptime": now - self.started_at,
            }
