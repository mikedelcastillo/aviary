"""On-demand camera snapshots for the ``/snapshot`` Telegram command.

Grabs each connected camera's most-recent decoded frame (published by the
capture loop into :class:`~lib.stats.CameraStats`) and writes it to the
persistent snapshots collection folder. The same files are sent back to the
requester as an album and are later swept into the annotation pipeline by
``import-collect-birds`` (the ``snapshots`` folder is imported by default).

This deliberately reuses the live frames the capture threads already decode
rather than opening a second RTSP session per camera: Tapo streams are flaky and
slow to handshake, so serving the in-flight frame is both faster and gentler on
the cameras than a fresh connect-per-snapshot would be.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from lib.collect import safe_path_component
from lib.stats import CameraStats


LOGGER = logging.getLogger("lib.snapshot")

# A snapshot serves the freshest decoded frame; on a healthy camera that is
# sub-second old. Past this the camera is stalled or disconnected and its last
# good frame is worth flagging in the album caption.
SNAPSHOT_STALE_SECONDS = 5.0


@dataclass
class CameraSnapshot:
    """One saved snapshot: which camera, where it landed, and how stale it was."""

    camera_name: str
    path: Path
    age_seconds: float | None


def snapshot_stem(camera_name: str) -> str:
    """``{utc}_{camera}_snapshot_{hash}`` — unique and sortable.

    Mirrors :func:`lib.collect.collection_stem` so snapshots sit beside collected
    detections and flow through ``import-collect-birds`` the same way; the
    ``snapshot`` token stands in for the detection label those files carry.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{safe_path_component(camera_name)}_snapshot_{uuid4().hex[:8]}"


def capture_snapshots(
    stats: dict[str, CameraStats],
    stats_lock: threading.Lock,
    output_dir: Path,
) -> list[CameraSnapshot]:
    """Save every camera's latest frame to ``output_dir``; return what was saved.

    Cameras without a frame yet (just discovered, never connected) are skipped.
    A single camera's write failure is logged and skipped so one bad frame never
    sinks the whole snapshot. The ``stats`` dict is copied under ``stats_lock``
    up front because the supervisor may add a camera from another thread, and
    the output directory is created lazily so an empty run leaves no folder.
    """
    with stats_lock:
        cameras = list(stats.items())

    saved: list[CameraSnapshot] = []
    made_dir = False
    for name, camera_stats in sorted(cameras, key=lambda item: item[0]):
        frame, age = camera_stats.latest_frame()
        if frame is None:
            continue
        if not made_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            made_dir = True
        path = output_dir / f"{snapshot_stem(name)}.jpg"
        try:
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"cv2.imwrite returned False for {path}")
        except Exception:
            LOGGER.exception("Failed to write snapshot for camera=%s", name)
            continue
        saved.append(CameraSnapshot(camera_name=name, path=path, age_seconds=age))

    return saved


def latest_frame_jpeg(camera_stats: CameraStats) -> bytes | None:
    """Encode a camera's most-recent decoded frame as JPEG bytes, or ``None``.

    Used by ``/find`` to attach proof photos without writing to disk. Returns
    ``None`` when the camera has never produced a frame or the encode fails.
    """
    frame, _ = camera_stats.latest_frame()
    if frame is None:
        return None
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buffer.tobytes()


def snapshot_caption(snap: CameraSnapshot) -> str:
    """Per-photo album caption: the camera name, flagged when the frame is stale."""
    if snap.age_seconds is not None and snap.age_seconds > SNAPSHOT_STALE_SECONDS:
        return f"{snap.camera_name} (stale, {snap.age_seconds:.0f}s ago)"
    return snap.camera_name
