"""Camera monitoring entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

from aviary_server.config import AppConfig, CameraConfig, ZoneConfig, _as_user_ids, build_config
from aviary_server.detector import BirdDetector, Detection, draw_detections
from aviary_server.telegram import TelegramNotifier, run_userinfo_bot


LOGGER = logging.getLogger("aviary_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default=os.getenv("AVIARY_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def apply_alert_zones(detections: list[Detection], zones: list[ZoneConfig]) -> list[Detection]:
    if not zones:
        return detections

    filtered: list[Detection] = []
    for detection in detections:
        for zone in zones:
            if point_in_polygon(detection.center, zone.polygon):
                detection.zone = zone.name
                if zone.alert:
                    filtered.append(detection)
                break
    return filtered


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
    alertable = apply_alert_zones(detections, camera.alert_zones)
    eligible = alert_state.eligible(camera.name, alertable)
    if not eligible:
        return

    snapshot_path = None
    if app_config.telegram.include_snapshot:
        snapshot_path = write_snapshot(app_config.snapshot_dir, camera.name, frame, eligible)

    labels = ", ".join(sorted({detection.label for detection in eligible}))
    LOGGER.info("Alerting camera=%s labels=%s snapshot=%s", camera.name, labels, snapshot_path)

    if notifier:
        notifier.send_detections(camera.name, eligible, snapshot_path)


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
    next_inference_at = 0.0

    while not stop_event.is_set():
        capture = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            LOGGER.warning("Could not open stream for %s; retrying in %.1fs", camera.name, camera.reconnect_seconds)
            stop_event.wait(camera.reconnect_seconds)
            continue

        LOGGER.info("Stream opened for %s", camera.name)
        try:
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    LOGGER.warning("Stream read failed for %s; reconnecting", camera.name)
                    break

                now = time.monotonic()
                if now < next_inference_at:
                    continue
                next_inference_at = now + min_frame_interval

                detections = detector.predict(frame)
                if detections:
                    handle_detections(app_config, camera, notifier, alert_state, frame, detections)
        except Exception:
            LOGGER.exception("Camera loop error for %s", camera.name)
            stop_event.wait(camera.reconnect_seconds)
        finally:
            capture.release()

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
