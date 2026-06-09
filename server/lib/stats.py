"""Thread-safe runtime statistics for camera workers."""

from __future__ import annotations

import threading
import time

from lib.objects import ObjectRegistry


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
    ) -> None:
        self.name = name
        self.sample_fps = sample_fps
        self._registry = registry
        self._fps_window = fps_window
        self._lock = threading.Lock()

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

    def record_inference(self, labels: list[str]) -> None:
        now = time.monotonic()
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
        if labels and self._registry is not None:
            self._registry.record(sorted(set(labels)), self.name)

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
