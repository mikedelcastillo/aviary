from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from lib.alerts import TELEGRAM_QUEUE_MAXSIZE, AlertDispatcher, TelegramJob
from lib.config import (
    AppConfig,
    CameraConfig,
    CameraCredentials,
    CollectConfig,
    DiscoveryConfig,
    FilterConfig,
    ModelConfig,
    TelegramConfig,
)
from lib.detector import Detection


class BlockingNotifier:
    """Stands in for a notifier stuck in a 429 backoff: send blocks until released."""

    def __init__(self, release: threading.Event) -> None:
        self._release = release
        self._lock = threading.Lock()
        self.sent = 0

    def send_detections(self, camera_name, detections, snapshot_path) -> None:
        self._release.wait(timeout=10.0)
        with self._lock:
            self.sent += 1


def _app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        snapshot_dir=tmp_path / "snapshots",
        model=ModelConfig(paths=(Path("/model.pt"),)),
        telegram=TelegramConfig(
            enabled=True, bot_token="token", user_ids=["1"], include_snapshot=True
        ),
        collect=CollectConfig(objects=frozenset({"bird"}), directory=tmp_path / "collect"),
        filter=FilterConfig(objects=frozenset()),
        credentials=CameraCredentials(username="user", password="pass"),
        discovery=DiscoveryConfig(),
    )


def _camera() -> CameraConfig:
    return CameraConfig(name="camera-1", enabled=True, rtsp_url="rtsp://example")


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_collection_continues_while_telegram_is_paused(tmp_path) -> None:
    release = threading.Event()
    notifier = BlockingNotifier(release)
    stop_event = threading.Event()
    dispatcher = AlertDispatcher(_app_config(tmp_path), notifier, stop_event, workers=2)

    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    detection = Detection(label="bird", confidence=0.9, bbox_xyxy=(1, 2, 11, 12))

    submitted = 10
    for _ in range(submitted):
        dispatcher.submit(_camera(), frame, [detection])

    collect_dir = tmp_path / "collect" / "bird"
    try:
        # Every submitted alert is collected even though the Telegram worker is
        # wedged on the very first send (still blocked, sent == 0).
        assert _wait_until(
            lambda: collect_dir.exists() and len(list(collect_dir.glob("*.jpg"))) == submitted
        )
        assert notifier.sent == 0

        # Releasing the pause lets the queued sends drain.
        release.set()
        assert _wait_until(lambda: notifier.sent >= 1)
    finally:
        release.set()
        stop_event.set()
        dispatcher.shutdown(timeout=2.0)


def test_enqueue_telegram_drops_oldest_when_full(tmp_path) -> None:
    stop_event = threading.Event()
    # notifier=None starts no Telegram worker, so nothing consumes the backlog
    # and the drop-oldest cap can be checked deterministically.
    dispatcher = AlertDispatcher(_app_config(tmp_path), None, stop_event, workers=1)
    try:
        camera = _camera()
        jobs = [
            TelegramJob(camera=camera, detections=[], snapshot_path=None)
            for _ in range(TELEGRAM_QUEUE_MAXSIZE + 5)
        ]
        for job in jobs:
            dispatcher._enqueue_telegram(job)

        # The queue is capped and the OLDEST five were dropped, leaving the most
        # recent maxsize jobs in order.
        assert dispatcher._telegram_queue.qsize() == TELEGRAM_QUEUE_MAXSIZE
        remaining = []
        while not dispatcher._telegram_queue.empty():
            remaining.append(dispatcher._telegram_queue.get_nowait())
        assert remaining == jobs[5:]
    finally:
        stop_event.set()
        dispatcher.shutdown(timeout=2.0)
