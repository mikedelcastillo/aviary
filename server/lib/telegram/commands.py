"""Telegram command responder for runtime bot commands."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from threading import Event, Thread

import requests

from lib.control import parse_duration
from lib.dashboard import STALE_FRAME_SECONDS
from lib.find import bird_last_seen
from lib.labels import pretty
from lib.objects import ObjectRegistry
from lib.stats import CameraStats
from lib.telegram.userinfo import parse_command
from lib.textfmt import render_telegram_html, to_plain


LOGGER = logging.getLogger("lib.telegram.commands")

StatusProvider = Callable[[], str]

# Descriptions for Telegram's slash-command menu — the list that pops up when a
# user types "/" in a chat with the bot. Registered at startup via
# ``setMyCommands`` so the app autocompletes the commands this bot answers. The
# order here is the menu's display order; only commands whose handler is wired
# up for a given run are actually registered (see ``run_command_bot``).
# Order here is the slash-menu display order (the user-requested order leads).
COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/activity": "Activity summary (e.g. /activity, /activity percy, /activity percy today)",
    "/sleep": "How the birds slept — sleep score + summary (e.g. /sleep, /sleep week)",
    "/care": "Bird-care guide (e.g. /care, /care diet, /care toxic, /care cockatiel)",
    "/discover": "Scan the local network for cameras",
    "/home": "Aim the pan-tilt cameras at their saved viewpoint",
    "/autofind": "Auto-search for missing birds (/autofind enable | disable)",
    "/stop": "Privacy mode: stop the cameras (optional duration, e.g. /stop 10m)",
    "/start": "Resume the cameras after a pause",
    "/status": "Show camera and detection status",
    "/find": "Find bird(s) across all cameras (e.g. /find percy, /find cockatiels, /find stop)",
    "/snapshot": "Capture a snapshot from every camera",
    "/pause": "Privacy mode: stop the cameras (alias of /stop)",
    "/play": "Resume the cameras (alias of /start)",
    "/resume": "Resume the cameras (alias of /start)",
    "/userinfo": "Show your Telegram user ID",
}

# Commands that turn privacy mode ON; the trailing text is parsed as a duration.
PAUSE_COMMANDS = ("/pause", "/stop")
# Commands that turn privacy mode OFF.
RESUME_COMMANDS = ("/play", "/resume", "/start")


def download_telegram_file(base_url: str, file_id: str, timeout: int = 30) -> bytes | None:
    """Fetch a Telegram file's bytes by file_id (getFile -> file download URL).

    ``base_url`` is ``https://api.telegram.org/bot<token>``; the binary lives
    under the parallel ``/file/bot<token>/<path>`` route. Returns None on any
    failure so the caller can degrade gracefully.
    """
    try:
        meta = requests.get(f"{base_url}/getFile", params={"file_id": file_id}, timeout=timeout)
        meta.raise_for_status()
        file_path = meta.json()["result"]["file_path"]
        file_base = base_url.replace("/bot", "/file/bot", 1)
        blob = requests.get(f"{file_base}/{file_path}", timeout=timeout)
        blob.raise_for_status()
        return blob.content
    except (requests.RequestException, KeyError, ValueError) as exc:
        LOGGER.warning("Failed to download Telegram file %s: %s", file_id, exc)
        return None


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


def _ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 60:
        return "just now"
    return f"{_format_duration(seconds)} ago"


def build_status_message(
    stats: dict[str, CameraStats],
    registry: ObjectRegistry | None,
    movement_alert_ratio: float = 0.0,
    camera_display: Callable[[str], str] = lambda name: name,
    *,
    known_birds: list[str] | None = None,
    ir_cameras: set[str] | None = None,
) -> str:
    """A compact, scannable status: which birds were last seen when, and each
    camera's health (with an IR/night marker).

    ``known_birds`` is the roster of individuals to always list (so a bird that
    has gone missing shows up as "not seen"); ``ir_cameras`` are camera ids
    currently in night/IR mode.
    """
    # Copy the mapping up front: the live ``stats`` dict is shared with the
    # camera supervisor, which adds entries from another thread.
    stats = dict(stats)
    snapshots = [stats[name].snapshot() for name in stats]
    object_rows = registry.snapshot() if registry is not None else []
    last_seen = bird_last_seen(object_rows)
    ir_cameras = ir_cameras or set()

    healthy = sum(
        1
        for snap in snapshots
        if snap["status"] == "connected"
        and (snap["since_frame"] is None or snap["since_frame"] <= STALE_FRAME_SECONDS)
    )

    lines = [f"🐦 **Aviary — {healthy}/{len(snapshots)} cameras healthy**", ""]

    # Birds, most-recently-seen first. Show EVERY label that has a sighting —
    # roster individuals AND any species/IR outline (e.g. "cockatiel" at night),
    # so /status isn't blank-looking during IR even though cameras are detecting.
    lines.append("**Birds — last seen:**")
    roster = known_birds if known_birds is not None else []
    shown = sorted(last_seen.items(), key=lambda kv: kv[1][0])
    for label, (since, camera) in shown:
        lines.append(f"  • {pretty(label)} — {_ago(since)} · {camera_display(camera)}")
    missing = [bird for bird in roster if bird not in last_seen]
    if missing:
        lines.append(f"  • not seen yet: {', '.join(pretty(b) for b in missing)}")
    if not shown and not missing:
        lines.append("  • nothing seen yet")

    # Cameras.
    lines.extend(["", "**Cameras:**"])
    if not snapshots:
        lines.append("  • none — send /discover")
    for snap in snapshots:
        fresh = snap["since_frame"] is not None and snap["since_frame"] <= STALE_FRAME_SECONDS
        if snap["status"] == "connected" and fresh:
            dot, word = "🟢", "live"
        elif snap["status"] == "connected":
            dot, word = "🟡", "stalled"
        else:
            dot, word = "🔴", str(snap["status"])
        ir = " · 🌙 IR" if snap["name"] in ir_cameras else ""
        lines.append(f"  {dot} {camera_display(snap['name'])} — {word}{ir} · {snap['fps']:.1f} fps")

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
    home_provider: Callable[[], str] | None = None,
    autofind_provider: Callable[[str], str] | None = None,
    snapshot_provider: Callable[[int], str] | None = None,
    pause_provider: Callable[[float | None], str] | None = None,
    resume_provider: Callable[[], str] | None = None,
    find_provider: Callable[[int, str], str] | None = None,
    nl_provider: Callable[[int, str], None] | None = None,
    photo_provider: Callable[[bytes], str] | None = None,
    activity_provider: Callable[[int, str], None] | None = None,
    sleep_provider: Callable[[int, str], None] | None = None,
    care_provider: Callable[[str], str] | None = None,
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
            ("/activity", activity_provider is not None),
            ("/sleep", sleep_provider is not None),
            ("/care", care_provider is not None),
            ("/discover", discover_provider is not None),
            ("/home", home_provider is not None),
            ("/autofind", autofind_provider is not None),
            ("/stop", pause_provider is not None),
            ("/start", resume_provider is not None),
            ("/status", status_provider is not None),
            ("/find", find_provider is not None),
            ("/snapshot", snapshot_provider is not None),
            ("/pause", pause_provider is not None),
            ("/play", resume_provider is not None),
            ("/resume", resume_provider is not None),
            ("/userinfo", True),
        )
        if present
    ]
    _register_bot_commands(base_url, available)

    LOGGER.info("Started Telegram command bot")

    def send(chat_id: int, text: str) -> None:
        """Best-effort sendMessage; a transient API failure must not kill polling.

        Renders as HTML (so **bold** markers show as real bold, never raw
        markdown) and, if Telegram rejects the HTML, retries once as plain text
        so a reply is never dropped over formatting."""
        try:
            requests.post(
                f"{base_url}/sendMessage",
                json={"chat_id": chat_id, "text": render_telegram_html(text), "parse_mode": "HTML"},
                timeout=15,
            ).raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("HTML send failed (%s); retrying plain", exc)
            try:
                requests.post(
                    f"{base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": to_plain(text)},
                    timeout=15,
                ).raise_for_status()
            except requests.RequestException as exc2:
                LOGGER.warning("Failed to send message: %s", exc2)

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
            # Guard every update: a handler raising (or a malformed update) must
            # never escape and kill this poll thread — that would silently take the
            # whole command bot offline until a restart. offset is already advanced,
            # so a poison update is logged and skipped, not re-fetched forever.
            try:
                message = update.get("message") or update.get("edited_message")
                if not message:
                    continue

                # A photo carries its text in "caption", not "text"; treat either as
                # the message text so a captioned photo's command/question still runs.
                text_in = message.get("text") or message.get("caption") or ""
                command = parse_command(text_in)

                user_id = (message.get("from") or {}).get("id")
                chat_id = (message.get("chat") or {}).get("id")
                if chat_id is None or user_id is None:
                    continue

                # A photo (with or without a caption): identify the birds + describe
                # it, for fun. Runs on its own thread (detector + VLM are slow) so
                # the poll loop keeps moving.
                photos = message.get("photo")
                if photos and photo_provider is not None:
                    if str(user_id) not in allowed:
                        send(chat_id, "Unauthorized.")
                        continue
                    file_id = photos[-1]["file_id"]  # largest size

                    def handle_photo(fid: str = file_id, cid: int = chat_id) -> None:
                        send(cid, "📷 Taking a look at your photo…")
                        image = download_telegram_file(base_url, fid)
                        if image is None:
                            send(cid, "Hmm, I couldn't download that photo.")
                            return
                        try:
                            reply = photo_provider(image)
                        except Exception:
                            LOGGER.exception("Photo analysis failed")
                            reply = "I couldn't make sense of that photo, sorry!"
                        send(cid, reply)

                    Thread(target=handle_photo, name="photo-analyze", daemon=True).start()
                    LOGGER.info("Handling photo from user %s", user_id)
                    # No caption -> done. With a caption, fall through so the caption
                    # is also handled as a command / natural-language request.
                    if not text_in.strip():
                        continue

                if command is None:
                    # Not a slash command: hand free text to the natural-language
                    # router (if enabled and the sender is allowed). It replies on
                    # its own background thread, so polling is never blocked.
                    if nl_provider is not None and text_in.strip() and str(user_id) in allowed:
                        try:
                            nl_provider(chat_id, text_in)
                        except Exception:
                            LOGGER.exception("Natural-language routing failed")
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

                if command == "/home":
                    if str(user_id) not in allowed or home_provider is None:
                        send(chat_id, "Unauthorized.")
                    else:
                        try:
                            send(chat_id, home_provider())
                        except Exception as exc:  # never let a PTZ error kill polling
                            LOGGER.exception("Home failed")
                            send(chat_id, f"Homing failed: {exc}")
                    LOGGER.info("Handled /home for user %s", user_id)
                    continue

                if command == "/autofind":
                    if str(user_id) not in allowed or autofind_provider is None:
                        send(chat_id, "Unauthorized.")
                    else:
                        try:
                            send(chat_id, autofind_provider(command_argument(text_in)))
                        except Exception as exc:
                            LOGGER.exception("Autofind toggle failed")
                            send(chat_id, f"Auto-find failed: {exc}")
                    LOGGER.info("Handled /autofind for user %s", user_id)
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

                if command == "/activity":
                    # Reads the day memory + summarises; can take a bird and/or
                    # "today" (/activity percy today). Backgrounded by the provider,
                    # so ack immediately.
                    if str(user_id) not in allowed or activity_provider is None:
                        send(chat_id, "Unauthorized.")
                    else:
                        send(chat_id, "📋 Looking back…")
                        try:
                            activity_provider(chat_id, command_argument(text_in))
                        except Exception as exc:  # never let it kill polling
                            LOGGER.exception("Activity failed")
                            send(chat_id, f"Activity failed: {exc}")
                    LOGGER.info("Handled /activity for user %s", user_id)
                    continue

                if command == "/sleep":
                    # How the flock slept: last night's score + summary, or the
                    # week trend with "/sleep week". Backgrounded by the provider.
                    if str(user_id) not in allowed or sleep_provider is None:
                        send(chat_id, "Unauthorized.")
                    else:
                        send(chat_id, "🌙 Checking how the birds slept…")
                        try:
                            sleep_provider(chat_id, command_argument(text_in))
                        except Exception as exc:  # never let it kill polling
                            LOGGER.exception("Sleep report failed")
                            send(chat_id, f"Sleep report failed: {exc}")
                    LOGGER.info("Handled /sleep for user %s", user_id)
                    continue

                if command == "/care":
                    # Bird-care guide from the knowledge base — fast + synchronous.
                    if str(user_id) not in allowed or care_provider is None:
                        send(chat_id, "Unauthorized.")
                    else:
                        try:
                            send(chat_id, care_provider(command_argument(text_in)))
                        except Exception as exc:  # never let it kill polling
                            LOGGER.exception("Care guide failed")
                            send(chat_id, f"Care info failed: {exc}")
                    LOGGER.info("Handled /care for user %s", user_id)
                    continue

                if command == "/find":
                    # Validate + launch on a background thread, then ack immediately.
                    # The search (up to 5 min) must never block this poll loop, so it
                    # pushes its own progress + result messages to the chat itself.
                    if str(user_id) not in allowed or find_provider is None:
                        send(chat_id, "Unauthorized.")
                    else:
                        target = command_argument(text_in)
                        try:
                            ack = find_provider(chat_id, target)
                        except Exception as exc:  # never let it kill polling
                            LOGGER.exception("Find failed")
                            ack = f"Find failed: {exc}"
                        send(chat_id, ack)
                    LOGGER.info("Handled /find for user %s", user_id)
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
                        duration = parse_duration(command_argument(text_in))
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
            except Exception:
                LOGGER.exception("Error handling update %s; skipping", update.get("update_id"))
