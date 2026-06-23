from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from lib.snapshot import (
    CameraSnapshot,
    capture_snapshots,
    snapshot_caption,
    snapshot_stem,
)
from lib.stats import CameraStats


def _frame(value: int = 7, shape=(8, 12, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


# --- CameraStats latest-frame store -----------------------------------------


def test_latest_frame_is_none_before_any_frame() -> None:
    frame, age = CameraStats("camera-1", 1.0).latest_frame()
    assert frame is None
    assert age is None


def test_latest_frame_returns_an_independent_copy() -> None:
    stats = CameraStats("camera-1", 1.0)
    stats.set_latest_frame(_frame(5))

    first, age = stats.latest_frame()
    assert first is not None
    assert age is not None and age >= 0.0
    # Mutating the returned copy must not corrupt the stored frame.
    first[:] = 99
    second, _ = stats.latest_frame()
    assert int(second.mean()) == 5


def test_set_latest_frame_replaces_previous() -> None:
    stats = CameraStats("camera-1", 1.0)
    stats.set_latest_frame(_frame(1))
    stats.set_latest_frame(_frame(2))
    frame, _ = stats.latest_frame()
    assert int(frame.mean()) == 2


# --- snapshot_stem / snapshot_caption ---------------------------------------


def test_snapshot_stem_sanitizes_camera_and_tags_snapshot() -> None:
    stem = snapshot_stem("camera-192.168.1.50")
    # Dots in the IP-derived name are sanitized to underscores, like collect.
    assert "camera-192_168_1_50" in stem
    assert "_snapshot_" in stem


def test_snapshot_caption_plain_when_fresh() -> None:
    snap = CameraSnapshot("camera-1", Path("x.jpg"), age_seconds=0.2)
    assert snapshot_caption(snap) == "camera-1"


def test_snapshot_caption_flags_stale_frame() -> None:
    snap = CameraSnapshot("camera-1", Path("x.jpg"), age_seconds=42.0)
    caption = snapshot_caption(snap)
    assert "camera-1" in caption
    assert "stale" in caption and "42s" in caption


# --- capture_snapshots ------------------------------------------------------


def test_capture_snapshots_writes_only_cameras_with_frames(tmp_path) -> None:
    with_frame = CameraStats("camera-2", 1.0)
    with_frame.set_latest_frame(_frame(3))
    without_frame = CameraStats("camera-1", 1.0)  # never received a frame
    stats = {"camera-2": with_frame, "camera-1": without_frame}

    out_dir = tmp_path / "snapshots"
    saved = capture_snapshots(stats, threading.Lock(), out_dir)

    assert len(saved) == 1
    entry = saved[0]
    assert entry.camera_name == "camera-2"
    assert entry.path.parent == out_dir
    assert entry.path.suffix == ".jpg"
    assert entry.path.exists()
    # The written file decodes back to the captured frame's shape.
    decoded = cv2.imread(str(entry.path))
    assert decoded.shape == (8, 12, 3)
    # Only the one camera produced a file.
    assert sorted(p.name for p in out_dir.glob("*.jpg")) == [entry.path.name]


def test_capture_snapshots_no_frames_creates_no_directory(tmp_path) -> None:
    stats = {"camera-1": CameraStats("camera-1", 1.0)}
    out_dir = tmp_path / "snapshots"

    saved = capture_snapshots(stats, threading.Lock(), out_dir)

    assert saved == []
    # Nothing to save => the folder is never created.
    assert not out_dir.exists()


def test_capture_snapshots_empty_stats(tmp_path) -> None:
    assert capture_snapshots({}, threading.Lock(), tmp_path / "snapshots") == []
