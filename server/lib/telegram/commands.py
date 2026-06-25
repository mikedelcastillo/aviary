"""Telegram command responder for runtime bot commands."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from threading import Event

import requests

from lib.control import parse_duration
from lib.dashboard import STALE_FRAME_SECONDS
from lib.objects import ObjectRegistry
from lib.stats import CameraStats
from lib.telegram.userinfo import parse_command


LOGGER = logging.getLogger("lib.telegram.commands")

StatusProvider = Callable[[], str]

# Descriptions for Telegram's slash-command menu — the list that pops up when a
# user types "/" in a chat with the bot. Registered at startup via
# ``setMyCommands`` so the app autocompletes the commands this bot answers. The
# order here is the menu's display order; only commands whose handler is wired
# up for a given run are actually registered (see ``run_command_bot``).
COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/status": "Show camera and detection status",
    "/snapshot": "Capture a snapshot from every camera",
    "/pause": "Privacy mode: stop the cameras (optional duration, e.g. /pause 10m)",
    "/stop": "Privacy mode: stop the cameras (alias of /pause)",
    "/play": "Resume the cameras after a pause",
    "/resume": "Resume the cameras after a pause (alias of /play)",
    "/discover": "Scan the local network for cameras",
    "/userinfo": "Show your Telegram user ID",
}

# Commands that turn privacy mode ON; the trailing text is parsed as a duration.
PAUSE_COMMANDS = ("/pause", "/stop")
# Commands that turn privacy mode OFF.
RESUME_COMMANDS = ("/play", "/resume")


def command_argument(text: str) -> str:
    """Return everything after the leading ``/command`` token, stripped.

    ``parse_command`` only yields the bare command; pause needs its duration
    argument ("/pause 10m" -> "10m"). An empty string means no argument was
    given (an indefinite pause).
    """
    parts = text.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_frame_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return _format_duration(seconds)


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _camera_status_text(snap: dict) -> str:
    status = str(snap["status"]).upper()
    if snap["status"] == "reconnecting" and snap["backoff"]:
        status += f" ({snap['backoff']:.0f}s backoff)"
    if (
        snap["status"] == "connected"
        and snap["since_frame"] is not None
        and snap["since_frame"] > STALE_FRAME_SECONDS
    ):
        return "STALLED"
    return status


def build_status_message(
    stats: dict[str, CameraStats],
    registry: ObjectRegistry | None,
    movement_alert_ratio: float,
) -> str:
    """Build a plain-text status message from dashboard data, excluding logs."""
    # Copy the mapping up front: the live ``stats`` dict is shared with the
    # camera supervisor, which adds entries from another thread. Iterating a
    # local copy makes this safe no matter how the caller obtained the dict
    # (the CameraStats values are themselves internally locked).
    stats = dict(stats)
    snapshots = [stats[name].snapshot() for name in stats]
    object_rows = registry.snapshot() if registry is not None else []

    total_frames = sum(snap["frames_total"] for snap in snapshots)
    total_detections = sum(snap["detections_total"] for snap in snapshots)
    total_alerts = sum(snap["alerts_sent"] for snap in snapshots)
    total_reconnects = sum(snap["reconnects"] for snap in snapshots)
    healthy = sum(
        1
        for snap in snapshots
        if snap["status"] == "connected"
        and (snap["since_frame"] is None or snap["since_frame"] <= STALE_FRAME_SECONDS)
    )

    lines = [
        "Aviary status",
        f"Cameras: {healthy}/{len(snapshots)} healthy",
        (
            "Totals: "
            f"{_format_count(total_frames)} frames, "
            f"{_format_count(total_detections)} detections, "
            f"{_format_count(total_alerts)} alerts, "
            f"{_format_count(total_reconnects)} reconnects"
        ),
        "",
        "Cameras",
    ]

    if not snapshots:
        lines.append("- none")
    for snap in snapshots:
        detection = snap["last_label"] or "none"
        if snap["last_label"]:
            detection += f" ({_format_duration(snap['since_detection'])} ago)"
        frame_age = _format_frame_age(snap["since_frame"])
        frame_text = "never" if frame_age == "never" else f"{frame_age} ago"
        lines.extend(
            [
                f"- {snap['name']}: {_camera_status_text(snap)}",
                f"  FPS: {snap['fps']:.2f} / {snap['sample_fps']:g}",
                f"  Last frame: {frame_text}; last detection: {detection}",
                (
                    "  Failures: "
                    f"{snap['consecutive_failures']}; "
                    f"uptime: {_format_duration(snap['uptime'])}"
                ),
            ]
        )

    lines.extend(["", "Objects"])
    if not object_rows:
        lines.append("- nothing seen yet")
    else:
        threshold_percent = movement_alert_ratio * 100
        for row in object_rows[:10]:
            movement_percent = row["movement_percent"]
            movement = "n/a" if movement_percent is None else f"{movement_percent:.1f}%"
            alert = (
                "never"
                if row["since_alert"] is None
                else f"{_format_duration(row['since_alert'])} ago"
            )
            moved_enough = (
                movement_percent is not None and movement_percent >= threshold_percent
            )
            flag = " alert-move" if moved_enough else ""
            lines.append(
                f"- {row['camera']} {row['label']}: "
                f"seen {_format_duration(row['since'])} ago, "
                f"alert {alert}, move {movement}{flag}, "
                f"count {_format_count(row['count'])}"
            )
        if len(object_rows) > 10:
            lines.append(f"- ...and {len(object_rows) - 10} more")

    return "\n".join(lines)


def _register_bot_commands(base_url: str, commands: list[str]) -> None:
    """Populate Telegram's slash-command menu so typing "/" autocompletes.

    Best-effort: ``setMyCommands`` failing must never stop the bot from polling,
    so a transient API error is logged and swallowed.
    """
    payload = []
    for command in commands:
        description = COMMAND_DESCRIPTIONS.get(command)
        if description is None:
            # A command with no menu description is a wiring mistake, not an API
            # failure; skip it so the rest of the menu still registers and the
            # bot still starts rather than dying on a KeyError here.
            LOGGER.warning("No menu description for %s; skipping", command)
            continue
        payload.append({"command": command.removeprefix("/"), "description": description})
    if not payload:
        # An empty list would *clear* the menu via setMyCommands, so don't call.
        return
    try:
        requests.post(
            f"{base_url}/setMyCommands",
            json={"commands": payload},
            timeout=15,
        ).raise_for_status()
        LOGGER.info("Registered %d Telegram bot command(s)", len(payload))
    except requests.RequestException as exc:
        LOGGER.warning("Failed to register bot commands: %s", exc)


def run_command_bot(
    bot_token: str,
    allowed_user_ids: list[str],
    status_provider: StatusProvider | None = None,
    stop_event: Event | None = None,
    poll_timeout_seconds: int = 30,
    discover_provider: Callable[[], str] | None = None,
    snapshot_provider: Callable[[int], str] | None = None,
    pause_provider: Callable[[float | None], str] | None = None,
    resume_provider: Callable[[], str] | None = None,
) -> None:
    """Long-poll Telegram and reply to supported bot commands."""
    base_url = f"https://api.telegram.org/bot{bot_token}"
    allowed = {str(user_id) for user_id in allowed_user_ids}
    offset: int | None = None

    # Advertise only the commands this run can actually answer. /userinfo is
    # always live; the rest depend on their provider being wired up. The order
    # follows COMMAND_DESCRIPTIONS so the menu is stable.
    available = [
        command
        for command, present in (
            ("/status", status_provider is not None),
            ("/snapshot", snapshot_provider is not None),
            ("/pause", pause_provider is not None),
            ("/stop", pause_provider is not None),
            ("/play", resume_provider is not None),
            ("/resume", resume_provider is not None),
            ("/discover", discover_provider is not None),
            ("/userinfo", True),
        )
        if present
    ]
    _register_bot_commands(base_url, available)

    LOGGER.info("Started Telegram command bot")

    def send(chat_id: int, text: str) -> None:
        """Best-effort sendMessage; a transient API failure must not kill polling."""
        try:
            requests.post(
                f"{base_url}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=15,
            ).raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Failed to send message: %s", exc)

    while stop_event is None or not stop_event.is_set():
        try:
            params: dict[str, int | str] = {"timeout": poll_timeout_seconds}
            if offset is not None:
                params["offset"] = offset
            response = requests.get(
                f"{base_url}/getUpdates",
                params=params,
                timeout=poll_timeout_seconds + 10,
            )
            response.raise_for_status()
            updates = response.json().get("result", [])
        except requests.RequestException as exc:
            LOGGER.warning("getUpdates failed: %s; retrying", exc)
            if stop_event is not None:
                stop_event.wait(5)
            else:
                time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message") or update.get("edited_message")
            if not message:
                continue

            command = parse_command(message.get("text", ""))
            if command is None:
                continue

            user_id = (message.get("from") or {}).get("id")
            chat_id = (message.get("chat") or {}).get("id")
            if chat_id is None or user_id is None:
                continue

            if command == "/discover":
                # Discovery sweeps the whole subnet and can take a few seconds,
                # so this is a two-message command: an immediate ack, then the
                # report once the scan returns. Handled inline (not via the
                # single shared send below) because of that two-step flow.
                if str(user_id) not in allowed or discover_provider is None:
                    send(chat_id, "Unauthorized.")
                else:
                    send(chat_id, "Scanning the local network for cameras...")
                    try:
                        report = discover_provider()
                    except Exception as exc:  # never let a scan error kill polling
                        LOGGER.exception("Discovery failed")
                        report = f"Discovery failed: {exc}"
                    send(chat_id, report)
                LOGGER.info("Handled /discover for user %s", user_id)
                continue

            if command == "/snapshot":
                # Like /discover, a two-message flow: grabbing every camera's
                # latest frame, saving, and uploading an album takes a moment, so
                # ack immediately, then deliver. The provider sends the album (it
                # owns the notifier + chat) and returns a final text summary; the
                # photos arrive between these two messages.
                if str(user_id) not in allowed or snapshot_provider is None:
                    send(chat_id, "Unauthorized.")
                else:
                    send(chat_id, "Capturing snapshots from all cameras...")
                    try:
                        report = snapshot_provider(chat_id)
                    except Exception as exc:  # never let a snapshot error kill polling
                        LOGGER.exception("Snapshot failed")
                        report = f"Snapshot failed: {exc}"
                    send(chat_id, report)
                LOGGER.info("Handled /snapshot for user %s", user_id)
                continue

            if command == "/userinfo":
                text = (
                    f"Your Telegram user ID is: {user_id}\n"
                    "Add it to TELEGRAM_USER_IDS to enable alerts and /status."
                )
            elif command == "/status":
                if str(user_id) not in allowed or status_provider is None:
                    text = "Unauthorized."
                else:
                    text = status_provider()
            elif command in PAUSE_COMMANDS:
                # /pause [duration] / /stop [duration]: enter privacy mode. The
                # argument is a casual duration ("10m", "1 hour"); absent/garbage
                # parses to None, i.e. an indefinite pause.
                if str(user_id) not in allowed or pause_provider is None:
                    text = "Unauthorized."
                else:
                    duration = parse_duration(command_argument(message.get("text", "")))
                    try:
                        text = pause_provider(duration)
                    except Exception as exc:  # never let it kill polling
                        LOGGER.exception("Pause failed")
                        text = f"Pause failed: {exc}"
            elif command in RESUME_COMMANDS:
                if str(user_id) not in allowed or resume_provider is None:
                    text = "Unauthorized."
                else:
                    try:
                        text = resume_provider()
                    except Exception as exc:  # never let it kill polling
                        LOGGER.exception("Resume failed")
                        text = f"Resume failed: {exc}"
            else:
                continue

            send(chat_id, text)
            LOGGER.info("Replied %s for user %s", command, user_id)
