"""Camera monitoring entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import threading
from collections.abc import Callable

from dotenv import load_dotenv

from lib.alerts import AlertDispatcher, AlertState
from lib.camera import configure_ffmpeg_capture
from lib.config import AppConfig, _as_user_ids, build_config
from lib.control import RuntimeControl
from lib.dashboard import Dashboard
from lib.detector import ObjectDetector
from lib.discovery import DiscoveryProgress
from lib.find import BirdFinder
from lib.ptz import PtzManager
from lib.objects import ObjectRegistry
from lib.snapshot import capture_snapshots, latest_frame_jpeg, snapshot_caption
from lib.stats import CameraStats
from lib.supervisor import CameraSupervisor, format_discovery_report
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
    discover_provider: Callable[[], str] | None = None,
    snapshot_provider: Callable[[int], str] | None = None,
    pause_provider: Callable[[float | None], str] | None = None,
    resume_provider: Callable[[], str] | None = None,
    find_provider: Callable[[int, str], str] | None = None,
) -> threading.Thread:
    """Run the Telegram command responder in a background daemon thread."""
    thread = threading.Thread(
        target=run_command_bot,
        args=(bot_token, user_ids, status_provider, stop_event),
        kwargs={
            "discover_provider": discover_provider,
            "snapshot_provider": snapshot_provider,
            "pause_provider": pause_provider,
            "resume_provider": resume_provider,
            "find_provider": find_provider,
        },
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

    # Start each run with an empty snapshots folder.
    if app_config.snapshot_dir.exists():
        shutil.rmtree(app_config.snapshot_dir)
    app_config.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # No cameras exist yet (they are discovered at runtime), so configure the
    # FFmpeg backend from CameraConfig defaults — every discovered camera shares
    # those, so the env option is identical once cameras appear.
    configure_ffmpeg_capture()

    # Load the model before the live dashboard takes over the screen; YOLO's
    # import/load chatter scrolls normally here instead of fighting the render.
    detector = ObjectDetector(app_config.model)
    notifier = build_notifier(app_config)
    alert_state = AlertState(
        app_config.telegram.last_seen_alert_seconds,
        app_config.telegram.bbox_movement_alert_ratio,
        app_config.filter.objects,
    )
    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    # Shared privacy/pause state. /pause (or "stop the cams") flips it on and
    # every camera thread releases its stream; /play flips it back. Created here
    # so the supervisor can hand it to each monitor thread and the command bot
    # can drive it.
    control = RuntimeControl()

    # The shared, dynamically-grown camera state. The supervisor is the sole
    # writer of `stats`; the dashboard render thread and the /status provider are
    # readers. Every read snapshots under `stats_lock` so a camera appearing
    # mid-render can never raise "dict changed size during iteration".
    registry = ObjectRegistry()
    stats: dict[str, CameraStats] = {}
    stats_lock = threading.Lock()
    # Shared live state of the current discovery sweep. The supervisor's workers
    # publish per-host progress into it; the dashboard reads it to render the
    # colour-coded discovery grid in place of the camera band while a scan runs.
    discovery_progress = DiscoveryProgress()

    # Delivery runs off the capture threads. The prepare stage is light (collect
    # + snapshot write), and the single Telegram worker is unaffected by this
    # pool size, so a small fixed pool is plenty regardless of how many cameras
    # discovery eventually finds — no need to scale it per camera.
    dispatcher = AlertDispatcher(app_config, notifier, stop_event, workers=4)

    supervisor = CameraSupervisor(
        app_config,
        detector,
        alert_state,
        dispatcher,
        registry,
        stats,
        stats_lock,
        stop_event,
        progress=discovery_progress,
        control=control,
    )

    # /userinfo stays available, /status exposes the runtime data, and /discover
    # re-runs the LAN sweep and starts any new cameras. The status provider must
    # snapshot the stats dict under the lock before reading it (the supervisor
    # may add a key at any time).
    def status_provider() -> str:
        with stats_lock:
            snap = dict(stats)
        message = build_status_message(
            snap, registry, app_config.telegram.bbox_movement_alert_ratio
        )
        # Lead with the privacy banner when paused so /status makes it obvious
        # why every camera reads "paused" and nothing is being recorded.
        if control.is_paused():
            return f"{control.status()}\n\n{message}"
        return message

    # /snapshot grabs every camera's latest live frame, saves it under the
    # persistent collect tree (so import-collect-birds later sweeps it into the
    # annotation pool), and uploads the set back to the requester as an album.
    # Saving beside collected detections — not the run-scoped snapshot_dir, which
    # is wiped each boot — is what makes the captures survive for training.
    snapshot_collect_dir = app_config.collect.directory / "snapshots"

    def snapshot_provider(chat_id: int) -> str:
        # Honour privacy mode: a paused server is consuming no frames, so refuse
        # rather than serve a stale pre-pause frame.
        if control.is_paused():
            return f"{control.status()} No snapshot while paused."
        saved = capture_snapshots(stats, stats_lock, snapshot_collect_dir)
        if not saved:
            return "No camera frames available yet; send /discover once cameras are online."
        if notifier is not None:
            items = [(snap.path.read_bytes(), snapshot_caption(snap)) for snap in saved]
            notifier.send_album(chat_id, items)
        return f"Sent {len(saved)} snapshot(s); saved to {snapshot_collect_dir}."

    # /find runs a background search (it pushes its own progress to the chat),
    # so it needs a notifier to talk to. Wired only when one exists.
    # PTZ manager discovers (and caches) which cameras are pan-tilt and builds a
    # patrol over the ones live right now; the search restores their facing after.
    ptz_manager = PtzManager(app_config.credentials)

    def grab_frame(camera_name: str) -> bytes | None:
        with stats_lock:
            camera_stats = stats.get(camera_name)
        return latest_frame_jpeg(camera_stats) if camera_stats is not None else None

    finder = (
        BirdFinder(
            registry,
            detector.known_labels,
            notify=notifier.send_text,
            grab_frame=grab_frame,
            send_album=notifier.send_album,
            make_patrol=lambda: ptz_manager.build_patrol(supervisor.active_hosts()),
        )
        if notifier is not None
        else None
    )

    def find_provider(chat_id: int, target: str) -> str:
        # Privacy first: a paused server is consuming no streams, so a search
        # would just stare at frozen registry state. Refuse instead.
        if control.is_paused():
            return f"{control.status()} Can't search while paused — /play first."
        assert finder is not None  # only wired when finder exists
        return finder.start(chat_id, target, stop_event)

    start_command_thread(
        app_config.telegram.bot_token,
        app_config.telegram.user_ids,
        stop_event,
        status_provider=status_provider,
        discover_provider=lambda: format_discovery_report(
            supervisor.discover_and_apply()
        ),
        snapshot_provider=snapshot_provider,
        pause_provider=control.pause,
        resume_provider=control.resume,
        find_provider=find_provider if finder is not None else None,
    )

    dashboard = Dashboard(
        stats,
        registry,
        log_level=getattr(logging, args.log_level.upper(), logging.INFO),
        logfile=os.getenv("AVIARY_LOG_FILE", "aviary.log"),
        movement_alert_ratio=app_config.telegram.bbox_movement_alert_ratio,
        stats_lock=stats_lock,
        discovery_progress=discovery_progress,
    )
    dashboard.start()

    # Initial sweep AFTER the dashboard is live so its discovery grid animates
    # the scan. (The heavy YOLO load already happened above and scrolled
    # normally.) Zero cameras is not fatal: cameras may still be booting, so we
    # warn and keep running — the user can /discover once they come up.
    LOGGER.info("Discovering cameras on the local network (initial sweep)...")
    initial = supervisor.discover_and_apply()
    if initial.added:
        LOGGER.info("Initial discovery started %d camera(s)", len(initial.added))
    else:
        LOGGER.warning(
            "Initial discovery found no cameras; send /discover once they are online"
        )

    try:
        # The supervisor owns the daemon monitor threads (each self-reconnects),
        # so the main thread just parks until a signal sets stop_event.
        while not stop_event.is_set():
            stop_event.wait(1.0)
    finally:
        dashboard.stop()
        supervisor.join()
        dispatcher.shutdown()


if __name__ == "__main__":
    main()
