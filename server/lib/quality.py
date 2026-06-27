"""Runtime RTSP stream quality selection.

Tapo exposes a high quality stream (``/stream1``) and a lower bandwidth stream
(``/stream2``). This controller lets commands force either path or run in
``auto`` mode, where each camera can independently fall back to stream2 when it
cannot keep up and promote to stream1 after a stable period.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from lib.config import CameraCredentials
from lib.discovery import build_rtsp_url

QUALITY_STREAM1 = "stream1"
QUALITY_STREAM2 = "stream2"
QUALITY_AUTO = "auto"
QUALITY_MODES = {QUALITY_STREAM1, QUALITY_STREAM2, QUALITY_AUTO}

STREAM_PATHS = {
    QUALITY_STREAM1: "/stream1",
    QUALITY_STREAM2: "/stream2",
}


@dataclass
class _CameraQuality:
    path: str = STREAM_PATHS[QUALITY_STREAM2]
    version: int = 0
    healthy_samples: int = 0
    weak_samples: int = 0
    changed_at: float = field(default_factory=time.monotonic)


class StreamQualityController:
    """Thread-safe quality mode and per-camera auto stream choice."""

    def __init__(
        self,
        credentials: CameraCredentials,
        *,
        port: int = 554,
        mode: str = QUALITY_STREAM1,
        promote_samples: int = 8,
        downgrade_samples: int = 3,
        promote_ratio: float = 0.90,
        downgrade_ratio: float = 0.60,
    ) -> None:
        if mode not in QUALITY_MODES:
            raise ValueError(f"unknown quality mode: {mode}")
        self._credentials = credentials
        self._port = port
        self._mode = mode
        self._version = 0
        self._promote_samples = max(1, promote_samples)
        self._downgrade_samples = max(1, downgrade_samples)
        self._promote_ratio = promote_ratio
        self._downgrade_ratio = downgrade_ratio
        self._lock = threading.RLock()
        self._cameras: dict[str, _CameraQuality] = {}

    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> str:
        mode = mode.strip().lower()
        if mode not in QUALITY_MODES:
            return "Usage: /quality stream1 | stream2 | auto"
        with self._lock:
            if mode == self._mode:
                return self.status()
            self._mode = mode
            self._version += 1
            forced_path = STREAM_PATHS.get(mode)
            for state in self._cameras.values():
                state.healthy_samples = 0
                state.weak_samples = 0
                if forced_path is not None and state.path != forced_path:
                    state.path = forced_path
                    state.version = self._version
                    state.changed_at = time.monotonic()
        return self.status()

    def register(self, host: str, preferred_path: str) -> None:
        with self._lock:
            if host in self._cameras:
                return
            path = STREAM_PATHS.get(self._mode, STREAM_PATHS[QUALITY_STREAM2])
            if self._mode == QUALITY_AUTO:
                path = STREAM_PATHS[QUALITY_STREAM2]
            elif preferred_path in set(STREAM_PATHS.values()):
                path = STREAM_PATHS[self._mode]
            self._cameras[host] = _CameraQuality(path=path, version=self._version)

    def selected(self, host: str) -> tuple[str, int]:
        with self._lock:
            state = self._cameras.setdefault(host, _CameraQuality())
            if self._mode in STREAM_PATHS:
                forced_path = STREAM_PATHS[self._mode]
                if state.path != forced_path:
                    self._version += 1
                    state.path = forced_path
                    state.version = self._version
                    state.changed_at = time.monotonic()
            return state.path, state.version

    def rtsp_url(self, host: str) -> tuple[str, str, int]:
        path, version = self.selected(host)
        return build_rtsp_url(self._credentials, host, self._port, path), path, version

    def observe(self, host: str, snap: dict, *, target_fps: float) -> None:
        with self._lock:
            if self._mode != QUALITY_AUTO:
                return
            state = self._cameras.setdefault(host, _CameraQuality())
            status = snap.get("status")
            fps = float(snap.get("fps") or 0.0)
            since_frame = snap.get("since_frame")
            failures = int(snap.get("consecutive_failures") or 0)
            target = max(0.1, target_fps)
            healthy = (
                status == "connected"
                and since_frame is not None
                and since_frame <= 10.0
                and fps >= target * self._promote_ratio
                and failures == 0
            )
            weak = (
                status != "connected"
                or since_frame is None
                or since_frame > 15.0
                or fps < target * self._downgrade_ratio
                or failures > 0
            )

            if healthy:
                state.healthy_samples += 1
                state.weak_samples = 0
            elif weak:
                state.weak_samples += 1
                state.healthy_samples = 0
            else:
                state.healthy_samples = 0
                state.weak_samples = 0

            if state.path == STREAM_PATHS[QUALITY_STREAM2]:
                if state.healthy_samples >= self._promote_samples:
                    self._switch_locked(state, STREAM_PATHS[QUALITY_STREAM1])
            elif state.weak_samples >= self._downgrade_samples:
                self._switch_locked(state, STREAM_PATHS[QUALITY_STREAM2])

    def _switch_locked(self, state: _CameraQuality, path: str) -> None:
        if state.path == path:
            return
        self._version += 1
        state.path = path
        state.version = self._version
        state.changed_at = time.monotonic()
        state.healthy_samples = 0
        state.weak_samples = 0

    def status(self) -> str:
        with self._lock:
            parts = [f"Quality mode: {self._mode}."]
            if self._cameras:
                selected = sorted(
                    f"{host}={state.path.removeprefix('/')}"
                    for host, state in self._cameras.items()
                )
                parts.append("Streams: " + ", ".join(selected) + ".")
            if self._mode == QUALITY_AUTO:
                parts.append("Auto promotes stable cameras to stream1 and falls back to stream2 on drops.")
            return "\n".join(parts)
