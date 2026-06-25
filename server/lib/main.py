"""Camera monitoring entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import threading
from collections.abc import Callable

import cv2
import numpy as np
from dotenv import load_dotenv

from lib.ai.chat import chat_reply
from lib.ai.client import OllamaClient
from lib.ai.intent import Intent
from lib.ai.memory import ConversationMemory
from lib.ai.router import NaturalLanguageRouter
from lib.ai.vlm import build_detection_context, describe_scene
from lib.activity_qa import ActivityResponder
from lib.alerts import AlertDispatcher, AlertState
from lib.camera import configure_ffmpeg_capture
from lib.camera_names import CameraNamer, name_cameras
from lib.config import AppConfig, _as_user_ids, build_config
from lib.console import (
    ConsoleDispatcher,
    ConsoleLogToggle,
    ConsoleNotifier,
    run_terminal_chat,
)
from lib.control import RuntimeControl, parse_duration
from lib.memory_maker import MemoryMaker
from lib.dashboard import Dashboard
from lib.detector import ObjectDetector
from lib.discovery import DiscoveryProgress
from lib.autofind import AutoFinder
from lib.find import BirdFinder
from lib.imaging import is_ir_frame
from lib.labels import pretty_labels
from lib.ptz import PtzManager
from lib.objects import ObjectRegistry
from lib.roster import load_sexes, load_species_members, pronoun_map, pronoun_sentence
from lib.snapshot import capture_snapshots, latest_frame_jpeg, snapshot_caption
from lib.stats import CameraStats
from lib.supervisor import CameraSupervisor, format_discovery_report
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
AUTO_DISCOVER_SECONDS = 30 * 60.0


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


def start_command_thread(
    bot_token: str,
    user_ids: list[str],
    stop_event: threading.Event,
    status_provider: Callable[[], str] | None = None,
    discover_provider: Callable[[], str] | None = None,
    home_provider: Callable[[], str] | None = None,
    autofind_provider: Callable[[str], str] | None = None,
    snapshot_provider: Callable[[int], str] | None = None,
    pause_provider: Callable[[float | None], str] | None = None,
    resume_provider: Callable[[], str] | None = None,
    find_provider: Callable[[int, str], str] | None = None,
    nl_provider: Callable[[int, str], None] | None = None,
    photo_provider: Callable[[bytes], str] | None = None,
    activity_provider: Callable[[int, str], None] | None = None,
) -> threading.Thread:
    """Run the Telegram command responder in a background daemon thread."""
    thread = threading.Thread(
        target=run_command_bot,
        args=(bot_token, user_ids, status_provider, stop_event),
        kwargs={
            "discover_provider": discover_provider,
            "home_provider": home_provider,
            "autofind_provider": autofind_provider,
            "snapshot_provider": snapshot_provider,
            "pause_provider": pause_provider,
            "resume_provider": resume_provider,
            "find_provider": find_provider,
            "nl_provider": nl_provider,
            "photo_provider": photo_provider,
            "activity_provider": activity_provider,
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
    home_provider=None,
    autofind_provider=None,
    activity_responder=None,
    memory=None,
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
        try:
            reply = chat_reply(client, app_config.ollama.llm_model, text, history=history)
        except Exception:
            LOGGER.exception("Chat reply failed")
            reply = "🤖 My language brain (Ollama) is unreachable right now."
        notifier.send_text(chat_id, reply or "🤔")
        if memory is not None:
            memory.record(chat_id, "user", text)
            memory.record(chat_id, "assistant", reply)

    def dispatch(chat_id: int, intent: Intent, text: str) -> None:
        action = intent.action
        if action == "pause":
            notifier.send_text(chat_id, control.pause(parse_duration(intent.argument)))
        elif action == "resume":
            notifier.send_text(chat_id, control.resume())
        elif action == "find":
            notifier.send_text(chat_id, find_provider(chat_id, intent.argument))
        elif action == "stop_find":
            notifier.send_text(chat_id, finder.stop_current())
        elif action == "discover":
            notifier.send_text(chat_id, "Scanning the local network for cameras...")
            notifier.send_text(chat_id, discover_provider())
        elif action == "home" and home_provider is not None:
            notifier.send_text(chat_id, home_provider())
        elif action == "autofind" and autofind_provider is not None:
            notifier.send_text(chat_id, autofind_provider(intent.argument))
        elif action == "status":
            notifier.send_text(chat_id, status_provider())
        elif action == "snapshot":
            notifier.send_text(chat_id, "Capturing snapshots from all cameras...")
            notifier.send_text(chat_id, snapshot_provider(chat_id))
        elif action == "activity" and activity_responder is not None:
            activity_responder.respond(chat_id, text, intent.argument)
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
    homed, total = ptz_manager.go_home(hosts)
    if total == 0:
        return "No pan-tilt cameras to home."
    return f"🏠 Sent {homed}/{total} pan-tilt camera(s) to their saved viewpoint."


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
        status_provider=status_provider,
        snapshot_provider=snapshot_text,
        home_provider=lambda: home_report(ptz_manager, supervisor.active_hosts()),
        activity_responder=activity_responder,
        memory=ConversationMemory() if ollama_client is not None else None,
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
        home_text=lambda: home_report(ptz_manager, supervisor.active_hosts()),
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
            snap,
            registry,
            app_config.telegram.bbox_movement_alert_ratio,
            camera_display=namer.display,
            known_birds=sorted(pronouns),
            ir_cameras=current_ir_cameras(),
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

    def current_ir_cameras() -> set[str]:
        """Camera ids whose latest frame is in night/IR (grayscale) mode."""
        with stats_lock:
            names = list(stats.keys())
        ir: set[str] = set()
        for name in names:
            jpeg = grab_frame(name)
            if jpeg and is_ir_frame(jpeg):
                ir.add(name)
        return ir

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
        if frame is not None:
            height, width = frame.shape[:2]
            context = build_detection_context(
                detector.predict(frame), width, height, member_species, pronouns
            )
        return describe_scene(
            ollama_client,
            app_config.ollama.vlm_model,
            image,
            context=context,
            timeout_seconds=VLM_DESCRIBE_TIMEOUT_SECONDS,
        )

    finder = (
        BirdFinder(
            registry,
            detector.known_labels,
            notify=notifier.send_text,
            grab_frame=grab_frame,
            send_photo=notifier.send_photo,
            describe_frame=describe_frame if ollama_client is not None else None,
            make_patrol=lambda: ptz_manager.build_patrol(supervisor.active_hosts()),
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

    def home_provider() -> str:
        return home_report(ptz_manager, supervisor.active_hosts())

    # Auto-find: search for birds missing >10 min while the cameras are in
    # daylight; auto-disables when all cameras go to night/IR.
    auto_finder = (
        AutoFinder(
            registry,
            finder,
            control,
            notifier,
            stop_event,
            ir_cameras=current_ir_cameras,
            camera_count=lambda: len(stats),
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

    def discover_provider() -> str:
        report = format_discovery_report(supervisor.discover_and_apply())
        # Face the saved viewpoint FIRST so the cameras are aimed right and the
        # naming below describes the home view, not wherever they were left.
        ptz_manager.go_home(supervisor.active_hosts())
        # Re-name every camera: new ones get named, and existing pan-tilt cameras
        # that have moved since last time get a fresh, accurate name.
        trigger_camera_naming(force=True)
        return report

    # Activity Q&A ("what did percy do today?") reads the collected-photos log;
    # conversation memory keeps ~20 turns per chat for coherent follow-ups.
    memory = ConversationMemory() if ollama_client is not None else None
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

    # Natural-language routing: free-text Telegram messages ("stop the cams",
    # "where's percy?") are classified by Ollama and dispatched to the same
    # providers as the slash commands. Wired only when a notifier exists (to
    # reply) and Ollama is enabled; degrades gracefully if Ollama is unreachable.
    nl_router = build_nl_router(
        app_config,
        notifier,
        finder,
        control,
        stop_event,
        ollama_client,
        find_provider=find_provider,
        discover_provider=discover_provider,
        status_provider=status_provider,
        snapshot_provider=snapshot_provider,
        home_provider=home_provider,
        autofind_provider=autofind_provider,
        activity_responder=activity_responder,
        memory=memory,
    )

    start_command_thread(
        app_config.telegram.bot_token,
        app_config.telegram.user_ids,
        stop_event,
        status_provider=status_provider,
        discover_provider=discover_provider,
        home_provider=home_provider,
        autofind_provider=autofind_provider if auto_finder is not None else None,
        snapshot_provider=snapshot_provider,
        pause_provider=control.pause,
        resume_provider=control.resume,
        find_provider=find_provider if finder is not None else None,
        nl_provider=nl_router.handle_async if nl_router is not None else None,
        photo_provider=photo_provider if ollama_client is not None else None,
        activity_provider=activity_provider if activity_responder is not None else None,
    )
    if auto_finder is not None:
        auto_finder.start()

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
        ).start()
        LOGGER.info(
            "Caretaker reports every ~%.0f min; raw photo alerts %s",
            app_config.memory_interval_minutes,
            "on" if app_config.raw_photo_alerts else "off",
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

    # Keep discovering in the background: two quick retries after boot (to catch
    # cameras whose RTSP was still busy from a previous run — common right after a
    # restart), then a silent sweep every AUTO_DISCOVER_SECONDS so cameras that
    # come online later are picked up without the user running /discover.
    # Idempotent — start_camera dedups by host.
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
        if notifier is not None:
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


if __name__ == "__main__":
    main()
