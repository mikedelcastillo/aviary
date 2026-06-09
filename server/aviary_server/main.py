"""Camera monitoring entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

from aviary_server.config import AppConfig, CameraConfig, _as_user_ids, build_config
from aviary_server.detector import BirdDetector, Detection, draw_detections
from aviary_server.telegram import TelegramNotifier, run_userinfo_bot


LOGGER = logging.getLogger("aviary_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default=os.getenv("AVIARY_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def snapshot_name(camera_name: str) -> str:
    safe_camera = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in camera_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{safe_camera}.jpg"


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


def write_snapshot(snapshot_dir: Path, camera_name: str, frame, detections: list[Detection]) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / snapshot_name(camera_name)
    annotated = draw_detections(frame, detections)
    ok = cv2.imwrite(str(path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to write snapshot: {path}")
    return path


def handle_detections(
    app_config: AppConfig,
    camera: CameraConfig,
    notifier: TelegramNotifier | None,
    alert_state: AlertState,
    frame,
    detections: list[Detection],
) -> None:
    eligible = alert_state.eligible(camera.name, detections)
    if not eligible:
        return

    snapshot_path = None
    if app_config.telegram.include_snapshot:
        snapshot_path = write_snapshot(app_config.snapshot_dir, camera.name, frame, eligible)

    labels = ", ".join(sorted({detection.label for detection in eligible}))
    LOGGER.info("Alerting camera=%s labels=%s snapshot=%s", camera.name, labels, snapshot_path)

    try:
        if notifier:
            notifier.send_detections(camera.name, eligible, snapshot_path)
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)


# Number of consecutive failed/empty reads tolerated before we tear the stream
# down and reconnect. Absorbs the occasional dropped frame without paying for a
# full RTSP re-handshake on every hiccup.
READ_FAILURE_LIMIT = 3


def configure_ffmpeg_capture(cameras: list[CameraConfig]) -> None:
    """Pin OpenCV's FFmpeg backend to TCP with a socket timeout.

    ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` is read by the FFmpeg backend each time a
    ``VideoCapture`` is constructed, so setting it once before the worker
    threads start covers every camera. TCP transport plus a bounded socket
    timeout stop a vanished Tapo camera from wedging a decode thread forever.
    """
    if not cameras:
        return
    transport = cameras[0].rtsp_transport
    # FFmpeg's ``timeout`` is the socket I/O timeout in microseconds.
    timeout_us = int(max(camera.read_timeout_seconds for camera in cameras) * 1_000_000)
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        f"rtsp_transport;{transport}|timeout;{timeout_us}",
    )


def open_capture(camera: CameraConfig) -> cv2.VideoCapture | None:
    """Open an RTSP capture with bounded open/read timeouts.

    Returns an opened capture, or ``None`` if the stream could not be opened
    (the caller backs off and retries). Timeout props are passed via the
    constructor params list and guarded with ``getattr`` so older OpenCV builds
    that lack them still work.
    """
    params: list[int] = []
    for prop_name, seconds in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", camera.open_timeout_seconds),
        ("CAP_PROP_READ_TIMEOUT_MSEC", camera.read_timeout_seconds),
    ):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            params += [prop, int(seconds * 1000)]

    try:
        capture = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG, params)
    except Exception:
        LOGGER.exception("Error opening stream for %s", camera.name)
        return None

    if not capture.isOpened():
        capture.release()
        return None

    # Keep the buffer shallow so inference runs on a fresh frame after any stall
    # instead of draining a backlog of stale ones.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def monitor_camera(
    app_config: AppConfig,
    camera: CameraConfig,
    detector: BirdDetector,
    notifier: TelegramNotifier | None,
    alert_state: AlertState,
    stop_event: threading.Event,
) -> None:
    LOGGER.info("Starting camera %s", camera.name)
    min_frame_interval = 1.0 / camera.sample_fps
    backoff = camera.reconnect_seconds

    while not stop_event.is_set():
        capture = open_capture(camera)
        if capture is None:
            LOGGER.warning("Could not open stream for %s; retrying in %.1fs", camera.name, backoff)
            stop_event.wait(backoff)
            backoff = min(backoff * 2, camera.max_reconnect_seconds)
            continue

        LOGGER.info("Stream opened for %s", camera.name)
        next_inference_at = 0.0
        consecutive_failures = 0
        try:
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= READ_FAILURE_LIMIT:
                        LOGGER.warning(
                            "Stream read failed for %s (%d consecutive); reconnecting",
                            camera.name,
                            consecutive_failures,
                        )
                        break
                    continue

                consecutive_failures = 0
                # A real frame arrived: the stream is healthy, so drop back to
                # the base reconnect delay for the next outage.
                backoff = camera.reconnect_seconds

                now = time.monotonic()
                if now < next_inference_at:
                    continue
                next_inference_at = now + min_frame_interval

                detections = detector.predict(frame)
                if detections:
                    handle_detections(app_config, camera, notifier, alert_state, frame, detections)
        except Exception:
            LOGGER.exception("Camera loop error for %s", camera.name)
        finally:
            capture.release()

        # The session ended unhealthily (stall, dropped stream, or decode
        # error). Pace before reconnecting and grow the delay while the camera
        # stays down; a healthy read above already reset it to the base.
        if not stop_event.is_set():
            stop_event.wait(backoff)
            backoff = min(backoff * 2, camera.max_reconnect_seconds)

    LOGGER.info("Stopped camera %s", camera.name)


def build_notifier(app_config: AppConfig) -> TelegramNotifier | None:
    if not app_config.telegram.enabled:
        return None
    return TelegramNotifier(app_config.telegram.bot_token, app_config.telegram.user_ids)


def install_signal_handlers(stop_event: threading.Event) -> None:
    def request_stop(signum, _frame) -> None:
        LOGGER.info("Received signal %s; stopping", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def start_userinfo_thread(bot_token: str, stop_event: threading.Event) -> threading.Thread:
    """Run the /userinfo responder in a background daemon thread.

    Active in every mode so operators can always collect the Telegram user ID
    of someone they want to add to TELEGRAM_USER_IDS, even while the detector
    is running.
    """
    thread = threading.Thread(
        target=run_userinfo_bot,
        args=(bot_token, stop_event),
        name="userinfo-bot",
        daemon=True,
    )
    thread.start()
    return thread


def main() -> None:
    args = parse_args()
    load_dotenv()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")

    # user_id mode: with only the bot token (no TELEGRAM_USER_IDS), run a
    # minimal bot that answers /userinfo and nothing else. No cameras or model
    # are loaded, and other missing env vars are ignored.
    if not _as_user_ids(os.environ.get("TELEGRAM_USER_IDS", "")):
        stop_event = threading.Event()
        install_signal_handlers(stop_event)
        start_userinfo_thread(bot_token, stop_event)
        while not stop_event.is_set():
            stop_event.wait(1.0)
        return

    app_config = build_config()
    enabled_cameras = [camera for camera in app_config.cameras if camera.enabled]
    if not enabled_cameras:
        raise SystemExit("No cameras are enabled in config")

    # Start each run with an empty snapshots folder.
    if app_config.snapshot_dir.exists():
        shutil.rmtree(app_config.snapshot_dir)
    app_config.snapshot_dir.mkdir(parents=True, exist_ok=True)

    configure_ffmpeg_capture(enabled_cameras)

    detector = BirdDetector(app_config.model)
    notifier = build_notifier(app_config)
    alert_state = AlertState(app_config.telegram.cooldown_seconds)
    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    # /userinfo stays available while the detector runs, so new viewers can be
    # added to the feed without restarting in user_id mode.
    start_userinfo_thread(app_config.telegram.bot_token, stop_event)

    with ThreadPoolExecutor(max_workers=len(enabled_cameras)) as executor:
        futures = [
            executor.submit(monitor_camera, app_config, camera, detector, notifier, alert_state, stop_event)
            for camera in enabled_cameras
        ]
        while not stop_event.is_set():
            for future in futures:
                if future.done():
                    future.result()
                    stop_event.set()
                    break
            stop_event.wait(1.0)


if __name__ == "__main__":
    main()
