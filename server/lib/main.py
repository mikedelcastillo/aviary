"""Camera monitoring entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from lib.alerts import AlertDispatcher, AlertState
from lib.camera import configure_ffmpeg_capture, monitor_camera
from lib.config import AppConfig, _as_user_ids, build_config
from lib.dashboard import Dashboard
from lib.detector import ObjectDetector
from lib.objects import ObjectRegistry
from lib.stats import CameraStats
from lib.telegram.commands import build_status_message, run_command_bot
from lib.telegram.notifier import TelegramNotifier


LOGGER = logging.getLogger("lib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default=os.getenv("AVIARY_LOG_LEVEL", "INFO"))
    return parser.parse_args()


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


def start_command_thread(
    bot_token: str,
    user_ids: list[str],
    stop_event: threading.Event,
    status_provider: Callable[[], str] | None = None,
) -> threading.Thread:
    """Run the Telegram command responder in a background daemon thread."""
    thread = threading.Thread(
        target=run_command_bot,
        args=(bot_token, user_ids, status_provider, stop_event),
        name="telegram-commands",
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

    # user_id mode: with only the bot token (no TELEGRAM_USER_IDS), keep the
    # command bot alive for /userinfo. No cameras or model are loaded, and
    # other missing env vars are ignored.
    if not _as_user_ids(os.environ.get("TELEGRAM_USER_IDS", "")):
        stop_event = threading.Event()
        install_signal_handlers(stop_event)
        start_command_thread(bot_token, [], stop_event)
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

    # Load the model before the live dashboard takes over the screen; YOLO's
    # import/load chatter scrolls normally here instead of fighting the render.
    detector = ObjectDetector(app_config.model)
    notifier = build_notifier(app_config)
    alert_state = AlertState(
        app_config.telegram.last_seen_alert_seconds,
        app_config.telegram.bbox_movement_alert_ratio,
    )
    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    # Delivery runs off the capture threads. One worker per camera means even a
    # worst-case 60s photo upload on every camera at once never makes one
    # camera's alert wait on another's.
    dispatcher = AlertDispatcher(app_config, notifier, stop_event, workers=len(enabled_cameras))

    registry = ObjectRegistry()
    stats = {
        camera.name: CameraStats(camera.name, camera.sample_fps, registry)
        for camera in enabled_cameras
    }

    # /userinfo stays available while the detector runs, and /status exposes
    # the same runtime data as the dashboard except for its event/log panel.
    start_command_thread(
        app_config.telegram.bot_token,
        app_config.telegram.user_ids,
        stop_event,
        status_provider=lambda: build_status_message(
            stats,
            registry,
            app_config.telegram.bbox_movement_alert_ratio,
        ),
    )

    dashboard = Dashboard(
        stats,
        registry,
        log_level=getattr(logging, args.log_level.upper(), logging.INFO),
        logfile=os.getenv("AVIARY_LOG_FILE", "aviary.log"),
        movement_alert_ratio=app_config.telegram.bbox_movement_alert_ratio,
    )
    dashboard.start()

    try:
        with ThreadPoolExecutor(max_workers=len(enabled_cameras)) as executor:
            futures = [
                executor.submit(
                    monitor_camera,
                    camera,
                    detector,
                    alert_state,
                    dispatcher,
                    stats[camera.name],
                    stop_event,
                )
                for camera in enabled_cameras
            ]
            while not stop_event.is_set():
                for future in futures:
                    if future.done():
                        future.result()
                        stop_event.set()
                        break
                stop_event.wait(1.0)
    finally:
        dashboard.stop()
        dispatcher.shutdown()


if __name__ == "__main__":
    main()
