"""Camera monitoring entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from collections.abc import Callable

import cv2
import numpy as np
from dotenv import load_dotenv

from lib.ai.chat import chat_reply
from lib.ai.client import OllamaClient
from lib.ai.context import build_chat_context, format_system_state
from lib.ai.intent import Intent
from lib.ai.memory import ConversationMemory
from lib.ai.router import NaturalLanguageRouter
from lib.ai.vlm import MAX_VLM_DIM, build_detection_context, describe_scene
from lib.activity_qa import ActivityResponder
from lib.alerts import AlertDispatcher, AlertState
from lib.camera import configure_ffmpeg_capture
from lib.camera_names import CameraNamer, name_cameras
from lib.care import care_reply, care_tip, toxic_food_in
from lib.care_scheduler import CareScheduler
from lib.clock import PH_TZ, now_ph
from lib.config import AppConfig, _as_user_ids, build_config
from lib.console import (
    ConsoleDispatcher,
    ConsoleLogToggle,
    ConsoleNotifier,
    run_terminal_chat,
)
from lib.control import RuntimeControl, parse_duration
from lib.memory_maker import MemoryMaker
from lib.dashboard import Dashboard, STALE_FRAME_SECONDS
from lib.detector import ObjectDetector
from lib.detection_log import DetectionLogger
from lib.durations import format_duration as _duration_text
from lib.weather import WeatherMonitor, fetch_forecast, has_location, summarize as summarize_weather
from lib.discovery import DiscoveryProgress
from lib.autofind import AutoFinder
from lib.find import BirdFinder, currently_visible, format_visible
from lib.imaging import downscale_array_to_jpeg
from lib.ir import IRState
from lib.labels import pretty_labels
from lib.ptz import PtzManager
from lib.quality import StreamQualityController
from lib.objects import ObjectRegistry
from lib.roster import load_sexes, load_species_members, pronoun_map, pronoun_sentence
from lib.snapshot import capture_snapshots, latest_frame_jpeg, snapshot_caption
from lib.sleep import SleepTracker, format_status_line
from lib.stats import CameraStats
from lib.suntimes import sun_times
from lib.supervisor import CameraSupervisor, format_discovery_progress, format_discovery_report
from lib.telegram.commands import build_status_message, run_command_bot
from lib.terminal_logging import NativeStderrRedirect
from lib.telegram.notifier import TelegramNotifier


LOGGER = logging.getLogger("lib")

# Upper bound on a single /find scene-description VLM call. Generous because a
# cold vision model under GPU contention (YOLO + a live search) can take well
# over a minute on its first call; the photo is already sent before this runs,
# so a long description never delays the find reply itself.
VLM_DESCRIBE_TIMEOUT_SECONDS = 150.0

# Silent background re-discovery cadence, so cameras that come online later get
# picked up without a manual /discover.
AUTO_DISCOVER_SECONDS = 10 * 60.0
CAMERA_ACTION_WAIT_SECONDS = 75.0
CAMERA_ACTION_POLL_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default=os.getenv("AVIARY_LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--chat",
        action="store_true",
        help="interactive terminal chat instead of the live dashboard",
    )
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
    # SIGHUP: a graceful stop too, so tearing down the background tmux session
    # (`server-down` → tmux kill-session) shuts the server down cleanly — flushing
    # the "stopping" Telegram broadcast — instead of an abrupt hangup kill.
    hup = getattr(signal, "SIGHUP", None)
    if hup is not None:
        signal.signal(hup, request_stop)


def start_command_thread(
    bot_token: str,
    user_ids: list[str],
    stop_event: threading.Event,
    status_provider: Callable[[], str] | None = None,
    discover_provider: Callable[..., str] | None = None,
    restart_provider: Callable[[], str] | None = None,
    detection_provider: Callable[[str], str] | None = None,
    home_provider: Callable[[], str] | None = None,
    quality_provider: Callable[[str], str] | None = None,
    autofind_provider: Callable[[str], str] | None = None,
    snapshot_provider: Callable[[int], str] | None = None,
    pause_provider: Callable[[float | None], str] | None = None,
    resume_provider: Callable[[], str] | None = None,
    find_provider: Callable[[int, str], str] | None = None,
    find_progress_message: Callable[[int | None], None] | None = None,
    nl_provider: Callable[[int, str], None] | None = None,
    photo_provider: Callable[[bytes], str] | None = None,
    activity_provider: Callable[[int, str], None] | None = None,
    sleep_provider: Callable[[int, str], None] | None = None,
    care_provider: Callable[[str], str] | None = None,
    weather_provider: Callable[[], str] | None = None,
) -> threading.Thread:
    """Run the Telegram command responder in a background daemon thread."""
    thread = threading.Thread(
        target=run_command_bot,
        args=(bot_token, user_ids, status_provider, stop_event),
        kwargs={
            "discover_provider": discover_provider,
            "restart_provider": restart_provider,
            "detection_provider": detection_provider,
            "home_provider": home_provider,
            "quality_provider": quality_provider,
            "autofind_provider": autofind_provider,
            "snapshot_provider": snapshot_provider,
            "pause_provider": pause_provider,
            "resume_provider": resume_provider,
            "find_provider": find_provider,
            "find_progress_message": find_progress_message,
            "nl_provider": nl_provider,
            "photo_provider": photo_provider,
            "activity_provider": activity_provider,
            "sleep_provider": sleep_provider,
            "care_provider": care_provider,
            "weather_provider": weather_provider,
        },
        name="telegram-commands",
        daemon=True,
    )
    thread.start()
    return thread


def build_nl_router(
    app_config: AppConfig,
    notifier,
    finder,
    control: RuntimeControl,
    stop_event: threading.Event,
    client: OllamaClient | None,
    *,
    find_provider,
    discover_provider,
    status_provider,
    snapshot_provider,
    restart_provider=None,
    home_provider=None,
    quality_provider=None,
    autofind_provider=None,
    activity_responder=None,
    memory=None,
    chat_context=None,
    sleep_provider=None,
    care_provider=None,
    weather_provider=None,
) -> NaturalLanguageRouter | None:
    """Wire the natural-language router to the command providers, or None.

    Returns None when the AI is disabled or there is no notifier to reply
    through. The dispatch maps each classified intent onto the very same
    providers the slash commands use, so behaviour (privacy refusals, acks) is
    identical whether the user types ``/find percy`` or "where's percy?".
    """
    if not app_config.ollama.enabled or notifier is None or finder is None or client is None:
        return None

    if client.is_available():
        LOGGER.info(
            "Ollama reachable at %s (llm=%s)",
            app_config.ollama.base_url,
            app_config.ollama.llm_model,
        )
    else:
        LOGGER.warning(
            "Ollama not reachable at %s; natural-language replies will say so",
            app_config.ollama.base_url,
        )

    def send_chat_reply(chat_id: int, text: str) -> None:
        history = memory.history(chat_id) if memory is not None else None
        # Ground the reply in live system state + relevant care knowledge so the
        # caretaker can actually answer questions (status, time, diet, sleep,
        # toxic foods) instead of only chit-chatting. Best-effort: a context
        # failure must never block the reply.
        context = None
        if chat_context is not None:
            try:
                context = chat_context(text)
            except Exception:
                LOGGER.exception("Building chat context failed")
        failed = False
        try:
            reply = chat_reply(
                client, app_config.ollama.llm_model, text, history=history, context=context
            )
        except Exception:
            LOGGER.exception("Chat reply failed")
            reply = "🤖 My language brain (Ollama) is unreachable right now."
            failed = True
        notifier.send_text(chat_id, reply or "🤔")
        # Don't persist a failed turn: a fake "I'm unreachable" assistant line
        # would be fed back as context on the next message and corrupt the chat.
        if memory is not None and not failed:
            memory.record(chat_id, "user", text)
            memory.record(chat_id, "assistant", reply)

    def dispatch(chat_id: int, intent: Intent, text: str) -> None:
        action = intent.action
        def update_or_send(message_id, reply: str) -> None:
            if (
                message_id is not None
                and hasattr(notifier, "edit_message_text")
                and notifier.edit_message_text(chat_id, message_id, reply)
            ):
                return
            notifier.send_text(chat_id, reply)

        # Safety first: a question naming a toxic food ("can percy eat avocado?")
        # must get the grounded care answer instead of "I haven't logged that" —
        # but ONLY when it was read as conversation/activity, never when it's a
        # real command (a "pause" that happens to mention a food must still pause).
        if action in ("chat", "activity") and toxic_food_in(text) is not None:
            send_chat_reply(chat_id, text)
            return
        if action == "pause":
            notifier.send_text(chat_id, control.pause(parse_duration(intent.argument)))
        elif action == "resume":
            notifier.send_text(chat_id, control.resume())
        elif action == "find":
            message_id = notifier.send_text(chat_id, find_provider(chat_id, intent.argument))
            if hasattr(finder, "attach_progress_message"):
                finder.attach_progress_message(message_id)
        elif action == "stop_find":
            notifier.send_text(chat_id, finder.stop_current())
        elif action == "discover":
            message_id = notifier.send_text(chat_id, "Scanning the local network for cameras...")
            def edit_discover(text: str) -> None:
                if message_id is not None and hasattr(notifier, "edit_message_text"):
                    notifier.edit_message_text(chat_id, message_id, text)

            # Debounce progress edits: discovery fires the callback ~twice per host
            # (~500x on a /24), and each edit is a blocking, rate-limited Telegram
            # call that would otherwise saturate the send gate and flood-control the
            # bot. Skip intermediate edits closer than 0.5s; the final report is
            # always sent below, unthrottled.
            last_edit = [0.0]

            def progress_update(text: str) -> None:
                now = time.monotonic()
                if now - last_edit[0] < 0.5:
                    return
                last_edit[0] = now
                edit_discover(text)

            edit_discover(discover_provider(progress_update))
        elif action == "restart" and restart_provider is not None:
            notifier.send_text(chat_id, restart_provider())
        elif action == "home" and home_provider is not None:
            notifier.send_text(chat_id, home_provider())
        elif action == "quality" and quality_provider is not None:
            notifier.send_text(chat_id, quality_provider(intent.argument))
        elif action == "autofind" and autofind_provider is not None:
            notifier.send_text(chat_id, autofind_provider(intent.argument))
        elif action == "status":
            notifier.send_text(chat_id, status_provider())
        elif action == "snapshot":
            message_id = notifier.send_text(chat_id, "Capturing snapshots from all cameras...")
            update_or_send(message_id, snapshot_provider(chat_id))
        elif action == "activity" and activity_responder is not None:
            activity_responder.respond(chat_id, text, intent.argument)
        elif action == "sleep" and sleep_provider is not None:
            sleep_provider(chat_id, intent.argument)
        elif action == "care" and care_provider is not None:
            notifier.send_text(chat_id, care_provider(intent.argument))
        elif action == "weather" and weather_provider is not None:
            notifier.send_text(chat_id, weather_provider())
        else:  # chat (or activity with no responder)
            send_chat_reply(chat_id, text)

    # Show a live "typing…" indicator while thinking when the notifier supports
    # it (Telegram does; the console no-ops).
    typing = (
        (lambda chat_id: notifier.send_chat_action(chat_id, "typing"))
        if hasattr(notifier, "send_chat_action")
        else None
    )
    return NaturalLanguageRouter(
        client,
        app_config.ollama.llm_model,
        finder.findable_labels,
        dispatch,
        notifier.send_text,
        typing=typing,
    )


def home_report(ptz_manager, hosts) -> str:
    """Send PTZ cameras to their saved viewpoint and report how many."""
    homed, total, without_preset = ptz_manager.go_home(hosts)
    if total == 0:
        return "No pan-tilt cameras to home."
    message = f"🏠 Sent {homed}/{total} pan-tilt camera(s) to their saved viewpoint."
    if without_preset:
        # A bare "0/2" is mystifying; tell the user WHY a camera didn't move.
        message += (
            f"\n⚠️ {without_preset} camera(s) have no saved home preset — "
            "set one in the Tapo app so I can aim them."
        )
    return message


def ready_camera_hosts(
    stats: dict[str, CameraStats],
    stats_lock: threading.Lock,
    *,
    max_frame_age: float = STALE_FRAME_SECONDS,
) -> set[str]:
    """Hosts whose capture loop is connected and publishing fresh frames."""
    with stats_lock:
        snapshots = [camera.snapshot() for camera in stats.values()]
    ready: set[str] = set()
    for snap in snapshots:
        since_frame = snap["since_frame"]
        if (
            snap["status"] == "connected"
            and since_frame is not None
            and since_frame <= max_frame_age
        ):
            ready.add(str(snap["name"]).removeprefix("camera-"))
    return ready


def wait_for_camera_action_window(
    *,
    discovery_progress: DiscoveryProgress,
    stats: dict[str, CameraStats],
    stats_lock: threading.Lock,
    stop_event: threading.Event,
    cancel_event: threading.Event | None = None,
    timeout_seconds: float = CAMERA_ACTION_WAIT_SECONDS,
    poll_seconds: float = CAMERA_ACTION_POLL_SECONDS,
) -> tuple[bool, str, set[str]]:
    """Wait until discovery is idle and at least one camera stream is fresh.

    PTZ requests are much more reliable once the camera has completed its RTSP
    reconnect and the app has seen a frame. The returned host set is the ready
    subset, so commands do not move cameras still marked connecting/reconnecting.
    """
    deadline = time.monotonic() + timeout_seconds
    saw_discovery = discovery_progress.is_active()
    saw_cameras_not_ready = False

    while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
        discovering = discovery_progress.is_active()
        if discovering:
            saw_discovery = True
        active = ready_camera_hosts(stats, stats_lock)
        all_hosts = set()
        with stats_lock:
            all_hosts = {
                str(camera.name).removeprefix("camera-") for camera in stats.values()
            }
        if not discovering and not all_hosts:
            return True, "", set()
        if not discovering and active:
            message = ""
            if saw_discovery:
                message = "Discovery finished; camera streams are live. Continuing."
            elif saw_cameras_not_ready:
                message = "Camera streams recovered. Continuing."
            return True, message, active
        if all_hosts and not active:
            saw_cameras_not_ready = True
        if time.monotonic() >= deadline:
            reason = (
                "Timed out waiting for discovery to finish."
                if discovering
                else "Timed out waiting for a live camera stream."
            )
            return False, reason, active
        time.sleep(poll_seconds)

    return False, "Cancelled before cameras were ready.", set()


def make_console_dispatcher(
    app_config: AppConfig,
    *,
    control: RuntimeControl,
    ollama_client: OllamaClient | None,
    detector: ObjectDetector,
    registry: ObjectRegistry,
    namer,
    member_species: dict[str, str],
    species_members: dict[str, tuple[str, ...]],
    pronouns: dict[str, str],
    memories_dir,
    grab_frame,
    describe_frame,
    ptz_manager,
    supervisor: CameraSupervisor,
    status_provider,
    discover_provider,
    restart_provider,
    detection_provider,
    home_provider,
    quality_provider,
    snapshot_text,
    stop_event: threading.Event,
) -> ConsoleDispatcher:
    """Build the terminal-chat dispatcher: same providers, replies to the console.

    A :class:`ConsoleNotifier` stands in for Telegram so the finder and NL router
    print to the terminal instead. Photos are never printed — find runs
    text-only (with the VLM description) here; images stay on Telegram.
    """

    def emit(text: str) -> None:
        print(f"\n{text}\n", flush=True)

    console_notifier = ConsoleNotifier(emit)

    console_finder = BirdFinder(
        registry,
        detector.known_labels,
        notify=console_notifier.send_text,
        grab_frame=grab_frame,
        send_photo=None,  # no images in the terminal
        describe_frame=describe_frame if ollama_client is not None else None,
        make_patrol=lambda: ptz_manager.build_patrol(supervisor.active_hosts()),
        camera_display=namer.display,
        species_members=species_members,
    )

    def console_find(chat_id: int, target: str) -> str:
        if target.strip().lower() in console_finder.STOP_WORDS:
            return console_finder.stop_current()
        if control.is_paused():
            return f"{control.status()} Can't search while paused — /play first."
        return console_finder.start(chat_id, target, stop_event)

    activity_responder = (
        ActivityResponder(
            memories_dir,
            ollama_client,
            app_config.ollama.llm_model,
            detector.known_labels,
            notify=console_notifier.send_text,
            send_album=None,
            find=lambda cid, arg: emit(console_find(cid, arg)),
            pronoun_note=pronoun_sentence(pronouns),
        )
        if ollama_client is not None
        else None
    )

    router = build_nl_router(
        app_config,
        console_notifier,
        console_finder,
        control,
        stop_event,
        ollama_client,
        find_provider=console_find,
        discover_provider=discover_provider,
        restart_provider=restart_provider,
        detection_provider=detection_provider,
        status_provider=status_provider,
        snapshot_provider=snapshot_text,
        home_provider=home_provider,
        quality_provider=quality_provider,
        activity_responder=activity_responder,
        memory=ConversationMemory() if ollama_client is not None else None,
        # The console has no live-state plumbing here, but care knowledge still
        # grounds diet/sleep/health questions in the terminal chat.
        chat_context=lambda text: build_chat_context(text, member_species=member_species),
    )

    def nl_handle(chat_id: int, text: str) -> None:
        if router is not None:
            router.handle_async(chat_id, text)
        else:
            emit("(AI is off — set OLLAMA_ENABLED=1; meanwhile use /commands.)")

    return ConsoleDispatcher(
        emit=emit,
        status_text=status_provider,
        discover_text=discover_provider,
        restart_text=restart_provider,
        detection_text=detection_provider,
        snapshot_text=snapshot_text,
        pause=control.pause,
        resume=control.resume,
        find=console_find,
        nl_handle=nl_handle,
        parse_duration=parse_duration,
        activity=(
            (lambda cid, arg: activity_responder.respond(cid, arg, arg))
            if activity_responder is not None
            else None
        ),
        home_text=home_provider,
        quality_text=quality_provider,
        toggle_logs=ConsoleLogToggle().toggle,
        on_quit=stop_event.set,
    )


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

    # One shared Ollama client for every AI feature (NL routing, chat, vision
    # scene descriptions, camera naming). None when the AI is disabled.
    ollama_client = (
        OllamaClient(
            app_config.ollama.base_url,
            timeout_seconds=app_config.ollama.timeout_seconds,
            vision_concurrency=app_config.ollama.vision_concurrency,
        )
        if app_config.ollama.enabled
        else None
    )

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

    # Friendly, VLM-derived display names per camera (identity stays IP-based).
    # Cached to disk so a restart shows the last known names instantly, before
    # the VLM has re-confirmed them.
    namer = CameraNamer(cache_path=app_config.collect.directory.parent / "camera_names.json")

    # The shared, dynamically-grown camera state. The supervisor is the sole
    # writer of `stats`; the dashboard render thread and the /status provider are
    # readers. Every read snapshots under `stats_lock` so a camera appearing
    # mid-render can never raise "dict changed size during iteration".
    registry = ObjectRegistry()
    # Per-camera IR/night state, stamped by each camera loop from the frame it
    # already decoded; /status, auto-find and the activity feed read it cheaply.
    ir_state = IRState()
    # Flock sleep tracker — assigned later (needs a notifier). The /status line
    # below reads it lazily, so the placeholder keeps the closure valid.
    sleep_tracker: SleepTracker | None = None
    stats: dict[str, CameraStats] = {}
    stats_lock = threading.Lock()
    # Shared live state of the current discovery sweep. The supervisor's workers
    # publish per-host progress into it; the dashboard reads it to render the
    # colour-coded discovery grid in place of the camera band while a scan runs.
    discovery_progress = DiscoveryProgress()
    quality_controller = StreamQualityController(
        app_config.credentials,
        port=app_config.discovery.rtsp_port,
        mode="stream1",
    )
    detection_logger = DetectionLogger(app_config.collect.directory.parent / "detection")

    # Delivery runs off the capture threads. The prepare stage is light (collect
    # + snapshot write), and the single Telegram worker is unaffected by this
    # pool size, so a small fixed pool is plenty regardless of how many cameras
    # discovery eventually finds — no need to scale it per camera.
    dispatcher = AlertDispatcher(app_config, notifier, stop_event, workers=4)
    restart_requested = threading.Event()

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
        ir_state=ir_state,
        quality=quality_controller,
        detection_logger=detection_logger,
    )

    # /userinfo stays available, /status exposes the runtime data, and /discover
    # re-runs the LAN sweep and starts any new cameras. The status provider must
    # snapshot the stats dict under the lock before reading it (the supervisor
    # may add a key at any time).
    def status_provider() -> str:
        with stats_lock:
            snap = dict(stats)
        now_utc = datetime.now(timezone.utc)
        logged_last_seen: dict[str, tuple[float, str]] = {}
        try:
            # Read today AND yesterday's (local-day) files so a sighting from
            # earlier this evening still surfaces across the local-midnight roll —
            # the most-recent (smallest "since") wins per label.
            for day in (now_utc - timedelta(days=1), now_utc):
                for row in detection_logger.activity_for_day(day):
                    if row.last_seen_at is None:
                        continue
                    since = max(0.0, (now_utc - row.last_seen_at).total_seconds())
                    current = logged_last_seen.get(row.label)
                    if current is None or since < current[0]:
                        logged_last_seen[row.label] = (since, row.camera)
        except Exception:
            LOGGER.exception("Reading detection log for /status failed")
        message = build_status_message(
            snap,
            registry,
            app_config.telegram.bbox_movement_alert_ratio,
            camera_display=namer.display,
            known_birds=sorted(pronouns),
            ir_cameras=ir_state.ir_cameras(),
            logged_last_seen=logged_last_seen,
        )
        # Lead with the privacy banner when paused so /status makes it obvious
        # why every camera reads "paused" and nothing is being recorded.
        if control.is_paused():
            return f"{control.status()}\n\n{message}"
        # While the flock is asleep, show the night-in-progress line.
        if sleep_tracker is not None:
            night_line = format_status_line(sleep_tracker.in_progress(), now_ph())
            if night_line:
                message = f"{message}\n\n{night_line}"
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
            items = [
                (snap.path.read_bytes(), snapshot_caption(snap, namer.display(snap.camera_name)))
                for snap in saved
            ]
            notifier.send_album(chat_id, items)
        return f"Sent {len(saved)} snapshot(s); saved to {snapshot_collect_dir}."

    # /find runs a background search (it pushes its own progress to the chat),
    # so it needs a notifier to talk to. Wired only when one exists.
    # PTZ manager discovers (and caches) which cameras are pan-tilt and builds a
    # patrol over the ones live right now; the search restores their facing after.
    ptz_manager = PtzManager(
        app_config.credentials,
        scan_cols=app_config.ptz_scan_cols,
        scan_rows=app_config.ptz_scan_rows,
    )

    def grab_frame(camera_name: str) -> bytes | None:
        with stats_lock:
            camera_stats = stats.get(camera_name)
        return latest_frame_jpeg(camera_stats) if camera_stats is not None else None


    # Roster groups (species -> members) for /find expansion, and the reverse
    # (member -> species) to annotate VLM detection context ("Pizza (a cockatiel)").
    species_members = load_species_members()
    member_species = {
        member: species for species, members in species_members.items() for member in members
    }
    # name -> "he"/"she" so descriptions use the right pronoun, plus a ground-
    # truth sentence ("Percy and Bambi are female …") for the summary LLM, which
    # works from captions that often don't carry pronouns.
    pronouns = pronoun_map(load_sexes())
    pronoun_note = pronoun_sentence(pronouns)
    # The caretaker's persistent day memory lives beside the collect tree.
    memories_dir = app_config.collect.directory.parent / "memories"

    def describe_frame(image: bytes) -> str | None:
        # Best-effort VLM scene description for /find hits. We FIRST run the
        # detector on the frame and hand the VLM exactly which birds are where —
        # otherwise it misses the small, distant birds and says "no birds".
        # Bounded timeout so a cold vision model never stalls the find reply.
        if ollama_client is None:
            return None
        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        context = None
        vlm_image = image
        if frame is not None:
            height, width = frame.shape[:2]
            context = build_detection_context(
                detector.predict(frame), width, height, member_species, pronouns
            )
            # Reuse the frame we just decoded: downscale the array directly so the
            # VLM call doesn't decode the JPEG a second time.
            vlm_image = downscale_array_to_jpeg(frame, MAX_VLM_DIM)
        return describe_scene(
            ollama_client,
            app_config.ollama.vlm_model,
            vlm_image,
            context=context,
            max_dim=None if frame is not None else MAX_VLM_DIM,
            timeout_seconds=VLM_DESCRIBE_TIMEOUT_SECONDS,
        )

    finder = (
        BirdFinder(
            registry,
            detector.known_labels,
            notify=notifier.send_text,
            edit_message=notifier.edit_message_text,
            grab_frame=grab_frame,
            send_photo=notifier.send_photo,
            describe_frame=describe_frame if ollama_client is not None else None,
            make_patrol=lambda: ptz_manager.build_patrol(
                ready_camera_hosts(stats, stats_lock)
            ),
            wait_until_ready=lambda stop, cancel: _wait_for_find_cameras(stop, cancel),
            camera_display=namer.display,
            species_members=species_members,
        )
        if notifier is not None
        else None
    )

    def photo_provider(image: bytes) -> str:
        # For fun: a user sends a photo, we run the detector to ID our birds and
        # the vision model (grounded by those detections) to describe it.
        if ollama_client is None:
            return "My vision brain (Ollama) is off right now."
        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return "That didn't look like an image I could read."
        detections = detector.predict(frame)
        height, width = frame.shape[:2]
        context = build_detection_context(detections, width, height, member_species, pronouns)
        try:
            description = describe_scene(
                ollama_client,
                app_config.ollama.vlm_model,
                image,
                context=context,
                timeout_seconds=VLM_DESCRIBE_TIMEOUT_SECONDS,
            )
        except Exception:
            LOGGER.exception("Photo description failed")
            description = ""
        birds = pretty_labels({d.label for d in detections})
        head = (
            f"📸 I spotted: {birds}!"
            if detections
            else "📸 I don't recognise any of our flock in this one."
        )
        return f"{head}\n{description}" if description else head

    def find_provider(chat_id: int, target: str) -> str:
        assert finder is not None  # only wired when finder exists
        # Stopping a search must work regardless of privacy state.
        if target.strip().lower() in finder.STOP_WORDS:
            return finder.stop_current()
        # Privacy first: a paused server is consuming no streams, so a search
        # would just stare at frozen registry state. Refuse instead.
        if control.is_paused():
            return f"{control.status()} Can't search while paused — /play first."
        return finder.start(chat_id, target, stop_event)

    def trigger_camera_naming(force: bool = False) -> None:
        # Name cameras from a live frame, on a background thread (the VLM is
        # slow). With force, RE-name already-named cameras too (a pan-tilt camera
        # may have moved). Only when the AI is enabled; otherwise cameras show
        # their IP.
        if ollama_client is None:
            return
        with stats_lock:
            cameras = list(stats.keys())
        threading.Thread(
            target=name_cameras,
            args=(namer, cameras, grab_frame, ollama_client, app_config.ollama.vlm_model),
            kwargs={
                "stop_event": stop_event,
                "timeout_seconds": VLM_DESCRIBE_TIMEOUT_SECONDS,
                "force": force,
                # Only re-VLM a camera that can move (pan-tilt); a fixed camera's
                # view is stable, so its cached name stands — saves cluster work.
                "is_movable": lambda cam: ptz_manager.camera_for(cam.removeprefix("camera-")) is not None,
            },
            name="camera-naming",
            daemon=True,
        ).start()

    def _wait_for_ready_cameras(
        stop: threading.Event,
        cancel: threading.Event | None = None,
    ) -> tuple[bool, str, set[str]]:
        return wait_for_camera_action_window(
            discovery_progress=discovery_progress,
            stats=stats,
            stats_lock=stats_lock,
            stop_event=stop,
            cancel_event=cancel,
        )

    def _wait_for_find_cameras(
        stop: threading.Event,
        cancel: threading.Event,
    ) -> tuple[bool, str]:
        ok, message, hosts = _wait_for_ready_cameras(stop, cancel)
        if not ok:
            return False, f"{message} Search skipped."
        if not hosts:
            return False, "No camera streams are live yet; send /discover once cameras are online."
        return True, message

    def home_provider() -> str:
        ok, message, hosts = _wait_for_ready_cameras(stop_event)
        if not ok:
            return f"{message}\nNo PTZ command was sent."
        report = home_report(ptz_manager, hosts)
        return f"{message}\n{report}" if message else report

    def quality_provider(argument: str) -> str:
        arg = argument.strip().lower()
        if not arg:
            return quality_controller.status()
        return quality_controller.set_mode(arg)

    def detection_provider(argument: str) -> str:
        tokens = [token.strip() for token in argument.split() if token.strip()]
        # "Today" and any explicit YYYY-MM-DD are the flock's LOCAL (Manila) day,
        # matching how the detection log now keys its files.
        day = datetime.now(PH_TZ)
        label_parts: list[str] = []
        for token in tokens:
            try:
                day = datetime.fromisoformat(token).replace(tzinfo=PH_TZ)
            except ValueError:
                label_parts.append(token)
        label = " ".join(label_parts).strip().lower() or None
        rows = detection_logger.activity_for_day(day, label=label)
        day_text = day.astimezone(PH_TZ).date().isoformat()
        if not rows:
            target = f" for {label}" if label else ""
            return f"No detection activity logged{target} on {day_text}."
        total = sum(row.total_seconds for row in rows)
        lines = [f"Detection activity on {day_text} — total {_duration_text(total)}:"]
        for row in rows[:30]:
            lines.append(
                f"  • {row.label} · {namer.display(row.camera)} — "
                f"{_duration_text(row.total_seconds)} ({row.observations} observations)"
            )
        if len(rows) > 30:
            lines.append(f"  • +{len(rows) - 30} more")
        lines.append(f"Saved in {app_config.collect.directory.parent / 'detection' / (day_text + '.json')}")
        return "\n".join(lines)

    def restart_provider() -> str:
        def request_restart() -> None:
            # Let the command handler send its acknowledgement before the main
            # thread starts shutting down the notifier and camera workers.
            time.sleep(0.5)
            LOGGER.info("Restart requested; re-execing current process")
            restart_requested.set()
            stop_event.set()

        threading.Thread(target=request_restart, name="restart-request", daemon=True).start()
        return "Restarting Aviary server..."

    # Auto-find: search for birds missing >10 min while the cameras are in
    # daylight; auto-disables when all cameras go to night/IR.
    auto_finder = (
        AutoFinder(
            registry,
            finder,
            control,
            notifier,
            stop_event,
            ir_cameras=ir_state.ir_cameras,
            camera_count=lambda: len(ir_state.known()),
            known_birds=sorted(pronouns),
        )
        if (finder is not None and notifier is not None)
        else None
    )

    def autofind_provider(argument: str) -> str:
        if auto_finder is None:
            return "Auto-find is unavailable (needs cameras + Telegram)."
        arg = argument.strip().lower()
        if arg in ("enable", "on", "start", "yes"):
            return auto_finder.set_enabled(True)
        if arg in ("disable", "off", "stop", "no"):
            return auto_finder.set_enabled(False)
        return auto_finder.status()

    def discover_provider(progress_update: Callable[[str], None] | None = None) -> str:
        def on_progress(progress: dict) -> None:
            if progress_update is not None:
                progress_update(format_discovery_progress(progress))

        report = format_discovery_report(supervisor.discover_and_apply(on_progress))
        # Face the saved viewpoint FIRST so the cameras are aimed right and the
        # naming below describes the home view, not wherever they were left.
        ok, wait_message, hosts = _wait_for_ready_cameras(stop_event)
        if ok and hosts:
            home = home_report(ptz_manager, hosts)
            report = f"{report}\n{wait_message}\n{home}" if wait_message else f"{report}\n{home}"
        elif not ok:
            report = f"{report}\n{wait_message}\nSkipped homing until cameras are responsive."
        # Re-name every camera: new ones get named, and existing pan-tilt cameras
        # that have moved since last time get a fresh, accurate name.
        trigger_camera_naming(force=True)
        return report

    # Activity Q&A ("what did percy do today?") reads the collected-photos log;
    # conversation memory keeps ~20 turns per chat for coherent follow-ups.
    memory = ConversationMemory() if ollama_client is not None else None
    def care_answer(text: str) -> str | None:
        # A grounded care answer when ``text`` is actually a care question, else
        # None — so a care Q routed to the activity path doesn't dead-end.
        if ollama_client is None:
            return None
        context = build_chat_context(text, member_species=member_species)
        if context is None:
            return None
        try:
            return chat_reply(ollama_client, app_config.ollama.llm_model, text, context=context)
        except Exception:
            LOGGER.exception("Care answer failed")
            return None

    activity_responder = (
        ActivityResponder(
            memories_dir,
            ollama_client,
            app_config.ollama.llm_model,
            detector.known_labels,
            notify=notifier.send_text,
            send_album=notifier.send_album,
            # A "what is X doing now?" with no logged memory kicks off a live
            # find; send its ack and let the search push its own photo + report.
            find=lambda cid, arg: notifier.send_text(cid, find_provider(cid, arg)),
            pronoun_note=pronoun_note,
            care_answer=care_answer,
        )
        if (ollama_client is not None and notifier is not None and finder is not None)
        else None
    )

    def activity_provider(chat_id: int, argument: str) -> None:
        # Backgrounded: reading the memory + an LLM summary takes a few seconds,
        # and must not block the Telegram poll loop. The responder sends the
        # summary + photos itself. The argument is used as both the bird filter
        # and the text (so "today" is detected from /activity percy today).
        if activity_responder is None:
            notifier.send_text(chat_id, "Activity memory is off (needs Ollama).")
            return
        threading.Thread(
            target=lambda: activity_responder.respond(chat_id, argument, argument),
            name="activity",
            daemon=True,
        ).start()

    # Flock sleep tracker: watches the IR light cycle to record each night's dark
    # window, scores it (10-12h target + consistency + disturbances), and answers
    # /sleep. Independent of the AI — only needs a notifier; the morning report is
    # env-gated off. The motion signal is the peak fresh-bird movement (the same
    # measure the caretaker's night-motion check uses).
    def _max_fresh_movement(fresh_seconds: float = 15.0) -> float:
        best = 0.0
        for row in registry.snapshot():
            since = row.get("since")
            if since is None or since > fresh_seconds:
                continue
            best = max(best, row.get("movement_percent") or 0.0)
        return best

    if notifier is not None:
        sleep_tracker = SleepTracker(
            notifier.broadcast_text,
            stop_event,
            all_ir=ir_state.all_ir,
            camera_count=lambda: len(ir_state.known()),
            movement=_max_fresh_movement,
            sleep_dir=app_config.collect.directory.parent / "sleep",
            morning_report=app_config.sleep_morning_report,
            client=ollama_client,
            llm_model=app_config.ollama.llm_model,
        )
        ir_state.add_listener(sleep_tracker.on_ir)
        sleep_tracker.start()
        LOGGER.info("Sleep tracker on (morning report %s)", app_config.sleep_morning_report)

    def sleep_provider(chat_id: int, argument: str) -> None:
        # Backgrounded (reads the night store off disk) so the poll loop is free.
        if sleep_tracker is None:
            notifier.send_text(chat_id, "Sleep tracking is unavailable.")
            return

        def work() -> None:
            from lib.sleep import NO_COVERAGE, format_last, format_week, sleep_streak

            arg = (argument or "").strip().lower()
            if arg in ("week", "trend", "7", "this week", "lately"):
                # recent() is newest-first; format_week wants oldest-first so the
                # trend sparkline reads left-to-right through time.
                notifier.send_text(chat_id, format_week(list(reversed(sleep_tracker.recent(7)))))
                return
            recent = sleep_tracker.recent(7)
            if recent:
                notifier.send_text(chat_id, format_last(recent[0], streak=sleep_streak(recent)))
                return
            in_progress = sleep_tracker.in_progress()
            line = format_status_line(in_progress, now_ph()) if in_progress is not None else None
            if line is not None:
                notifier.send_text(chat_id, f"{line} — I'll have a full report after they wake. 🌙")
            elif len(ir_state.known()) == 0:
                notifier.send_text(chat_id, NO_COVERAGE)
            else:
                notifier.send_text(chat_id, "No finished nights tracked yet — I'll report after the birds' first full night. 🌙")

        threading.Thread(target=work, name="sleep", daemon=True).start()

    def care_provider(argument: str) -> str:
        # The /care guide — pure + fast, so it answers inline (no thread/AI).
        return care_reply(argument, member_species=member_species)

    # Natural-language routing: free-text Telegram messages ("stop the cams",
    # "where's percy?") are classified by Ollama and dispatched to the same
    # providers as the slash commands. Wired only when a notifier exists (to
    # reply) and Ollama is enabled; degrades gracefully if Ollama is unreachable.
    def chat_context_provider(text: str) -> str | None:
        # Live system state for grounded chat answers ("how are things?", "is it
        # dark yet?", "are the cameras ok?"). Snapshots the shared stats under its
        # lock; the care knowledge is folded in by build_chat_context.
        with stats_lock:
            snaps = [camera.snapshot() for camera in stats.values()]
        healthy = sum(
            1
            for snap in snaps
            if snap["status"] == "connected"
            and (snap["since_frame"] is None or snap["since_frame"] <= STALE_FRAME_SECONDS)
        )
        if ir_state.all_ir():
            daylight = "night"
        elif ir_state.ir_cameras():
            daylight = "mixed"
        else:
            daylight = "day"
        visible = currently_visible(registry.snapshot(), 15.0)
        system_state = format_system_state(
            now_ph(),
            paused=control.is_paused(),
            pause_status=control.status(),
            cameras_total=len(snaps),
            cameras_healthy=healthy,
            visible_text=format_visible(visible, namer.display) if visible else "",
            daylight=daylight,
            autofind_on=auto_finder.enabled if auto_finder is not None else None,
        )
        return build_chat_context(text, system_state=system_state, member_species=member_species)

    # /weather: today's outlook + bird-safety advice, fetched on demand. Stays
    # None (command hidden) until a location is configured in .env, so we never
    # query the 0,0 placeholder.
    weather_provider = None
    if notifier is not None and has_location(app_config.latitude, app_config.longitude):
        def _weather_summary() -> str:
            forecast = fetch_forecast(app_config.latitude, app_config.longitude)
            if forecast is None:
                return "Couldn't reach the weather service just now — try again shortly."
            return summarize_weather(
                forecast,
                hot_c=app_config.weather_hot_c,
                cold_c=app_config.weather_cold_c,
            )

        weather_provider = _weather_summary

    nl_router = build_nl_router(
        app_config,
        notifier,
        finder,
        control,
        stop_event,
        ollama_client,
        find_provider=find_provider,
        discover_provider=discover_provider,
        restart_provider=restart_provider,
        status_provider=status_provider,
        snapshot_provider=snapshot_provider,
        home_provider=home_provider,
        quality_provider=quality_provider,
        autofind_provider=autofind_provider,
        activity_responder=activity_responder,
        memory=memory,
        chat_context=chat_context_provider,
        sleep_provider=sleep_provider if sleep_tracker is not None else None,
        care_provider=care_provider,
        weather_provider=weather_provider,
    )

    start_command_thread(
        app_config.telegram.bot_token,
        app_config.telegram.user_ids,
        stop_event,
        status_provider=status_provider,
        discover_provider=discover_provider,
        restart_provider=restart_provider,
        detection_provider=detection_provider,
        home_provider=home_provider,
        quality_provider=quality_provider,
        autofind_provider=autofind_provider if auto_finder is not None else None,
        snapshot_provider=snapshot_provider,
        pause_provider=control.pause,
        resume_provider=control.resume,
        find_provider=find_provider if finder is not None else None,
        find_progress_message=finder.attach_progress_message if finder is not None else None,
        nl_provider=nl_router.handle_async if nl_router is not None else None,
        photo_provider=photo_provider if ollama_client is not None else None,
        activity_provider=activity_provider if activity_responder is not None else None,
        sleep_provider=sleep_provider if sleep_tracker is not None else None,
        care_provider=care_provider,
        weather_provider=weather_provider,
    )
    if auto_finder is not None:
        auto_finder.start()

    # Weather watch: poll the forecast several times a day and warn on a hot day
    # or a cold night for the flock. Needs a notifier + a configured location.
    if (
        app_config.weather_alerts
        and notifier is not None
        and has_location(app_config.latitude, app_config.longitude)
    ):
        WeatherMonitor(
            notifier.broadcast_text,
            stop_event,
            lambda: fetch_forecast(app_config.latitude, app_config.longitude),
            hot_c=app_config.weather_hot_c,
            cold_c=app_config.weather_cold_c,
        ).start()
        LOGGER.info(
            "Weather alerts on (hot>=%.0f°C, cold<=%.0f°C)",
            app_config.weather_hot_c,
            app_config.weather_cold_c,
        )

    # The caretaker: smart activity reports (immediate on new birds, edit-in-place
    # otherwise) + persistent day memory under data/server/memories. Only when the
    # AI is on, there's someone to send to, and a beat is set.
    memories_dir = app_config.collect.directory.parent / "memories"
    if (
        ollama_client is not None
        and notifier is not None
        and app_config.memory_interval_minutes > 0
    ):
        MemoryMaker(
            memories_dir,
            registry,
            grab_frame,
            describe_frame,
            ollama_client,
            app_config.ollama.llm_model,
            notifier,
            stop_event,
            interval_seconds=app_config.memory_interval_minutes * 60.0,
            camera_display=namer.display,
            pronoun_note=pronoun_note,
            night_mode=ir_state.all_ir,
        ).start()
        LOGGER.info(
            "Caretaker reports every ~%.0f min; raw photo alerts %s",
            app_config.memory_interval_minutes,
            "on" if app_config.raw_photo_alerts else "off",
        )

    # Daily/weekly care reminders (fresh food + water, produce pickup, midday
    # enrichment, wind-down, bedtime, weekly clean + weigh) keyed to the observed
    # light (cameras leaving/entering IR) and PH sunrise/sunset. Independent of
    # the AI — it only needs a notifier to talk to.
    if notifier is not None and app_config.care_reminders:
        def last_night_sleep() -> str | None:
            # Last night's sleep one-liner folded into the morning nudge — only
            # when it's genuinely LAST night (its evening was yesterday) and
            # already finalized, never a stale older night.
            if sleep_tracker is None:
                return None
            night = sleep_tracker.last_finalized()
            if night is None or night.night_of != now_ph().date() - timedelta(days=1):
                return None
            from lib.sleep import format_morning

            return format_morning(night)

        CareScheduler(
            notifier.broadcast_text,
            stop_event,
            all_ir=ir_state.all_ir,
            camera_count=lambda: len(ir_state.known()),
            sun_times=lambda day: sun_times(
                day, app_config.latitude, app_config.longitude, app_config.utc_offset_hours
            ),
            sleep_summary=last_night_sleep,
            weekly_tip=lambda: care_tip(now_ph().isocalendar()[1]),
            state_path=app_config.collect.directory.parent / "care_scheduler.json",
        ).start()
        LOGGER.info(
            "Care reminders on (lat=%.4f lon=%.4f)", app_config.latitude, app_config.longitude
        )

    # Presentation: the live dashboard by default, or the interactive terminal
    # chat with --chat. In chat mode we keep the terminal clean — Python logs go
    # to the logfile and native ffmpeg/h264 stderr is redirected there too (toggle
    # live logs from inside the chat with /logs).
    dashboard = None
    stderr_redirect = None
    if args.chat:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        file_handler = logging.FileHandler(os.getenv("AVIARY_LOG_FILE", "aviary.log"))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(file_handler)
        stderr_redirect = NativeStderrRedirect(os.getenv("AVIARY_LOG_FILE", "aviary.log"))
        stderr_redirect.start()
    else:
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
    # Name cameras from a live frame (background, VLM). force=True so cached
    # names loaded from disk get re-confirmed now that the machine is up — a
    # camera may have moved since last run — rather than only on the 30-min sweep.
    trigger_camera_naming(force=True)

    # Keep discovering in the background: a silent sweep every
    # AUTO_DISCOVER_SECONDS so IP changes are picked up without the user running
    # /discover. Reconciliation happens only after each scan returns, so
    # unchanged cameras keep running with no drop while discovery is in flight.
    def _rediscover(force: bool = False) -> None:
        try:
            applied = supervisor.discover_and_apply()
        except Exception:
            LOGGER.exception("Auto-discovery sweep failed")
            return
        if applied.added:
            LOGGER.info("Auto-discovery started %d more camera(s)", len(applied.added))
        # On the periodic sweep, re-home then re-name every camera (force) since
        # pan-tilt cameras may have moved; the quick post-boot retries just name
        # new ones.
        if applied.added or force:
            ptz_manager.go_home(supervisor.active_hosts())
            trigger_camera_naming(force=force)

    def _discovery_background() -> None:
        # Two quick post-boot retries catch a camera whose RTSP socket is still
        # held by the previous process run (e.g. just after a /restart) before the
        # first in-sweep probe — too slow for the periodic 10-minute sweep alone.
        for delay in (15.0, 45.0):
            if stop_event.wait(delay):
                return
            _rediscover()
        while not stop_event.wait(AUTO_DISCOVER_SECONDS):
            _rediscover(force=True)

    threading.Thread(target=_discovery_background, name="auto-discovery", daemon=True).start()

    # Announce we're live (with the camera count) so the user knows monitoring
    # has resumed — e.g. after a restart. Best-effort; never block startup.
    if notifier is not None:
        camera_count = len(initial.added)
        try:
            notifier.broadcast_text(
                f"🟢 Aviary server started — watching {camera_count} "
                f"camera{'s' if camera_count != 1 else ''}."
            )
        except Exception:
            LOGGER.exception("Server-started broadcast failed")

    try:
        if args.chat:
            def snapshot_text(_chat_id: int) -> str:
                if control.is_paused():
                    return f"{control.status()} No snapshot while paused."
                saved = capture_snapshots(stats, stats_lock, snapshot_collect_dir)
                if not saved:
                    return "No camera frames yet; /discover once cameras are online."
                return (
                    f"Captured {len(saved)} snapshot(s) → {snapshot_collect_dir} "
                    "(photos are saved + sent to Telegram, not shown here)."
                )

            console_dispatcher = make_console_dispatcher(
                app_config,
                control=control,
                ollama_client=ollama_client,
                detector=detector,
                registry=registry,
                namer=namer,
                member_species=member_species,
                species_members=species_members,
                pronouns=pronouns,
                memories_dir=memories_dir,
                grab_frame=grab_frame,
                describe_frame=describe_frame,
                ptz_manager=ptz_manager,
                supervisor=supervisor,
                status_provider=status_provider,
                discover_provider=discover_provider,
                restart_provider=restart_provider,
                detection_provider=detection_provider,
                home_provider=home_provider,
                quality_provider=quality_provider,
                snapshot_text=snapshot_text,
                stop_event=stop_event,
            )
            run_terminal_chat(console_dispatcher, stop_event)
        else:
            # The supervisor owns the daemon monitor threads (each self-reconnects),
            # so the main thread just parks until a signal sets stop_event.
            while not stop_event.is_set():
                stop_event.wait(1.0)
    finally:
        # Tell the user we're going down BEFORE the slow joins, while the
        # notifier still works. Only on a graceful stop (SIGINT/SIGTERM set
        # stop_event); a SIGKILL can't be announced. Best-effort.
        if notifier is not None and not restart_requested.is_set():
            try:
                notifier.broadcast_text("🔴 Aviary server stopping — cameras going offline.")
            except Exception:
                LOGGER.exception("Server-stopping broadcast failed")
        if dashboard is not None:
            dashboard.stop()
        if stderr_redirect is not None:
            stderr_redirect.stop()
        supervisor.join()
        dispatcher.shutdown()
        if restart_requested.is_set():
            os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
