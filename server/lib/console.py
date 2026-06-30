"""Interactive terminal chat for the server (``uv run server --chat``).

A Claude-Code-style prompt to talk to the running aviary: type a slash command
(``/status``, ``/find percy``, ``/pause 10m`` …) OR plain language ("where's
percy?", "what did matcha do today?") and get a text reply right in the
terminal. It drives the SAME control/finder/AI providers as the Telegram bot, so
behaviour is identical — only the replies land in your console instead of a chat.

Images never print to the terminal (they go to Telegram); a photo result shows a
short "[sent N photo(s) to Telegram]" note instead. Server logs are hidden by
default and toggled with ``/logs`` so the conversation stays readable. The camera
status boxes are shown with ``/status`` (or ``/cams``).

The :class:`ConsoleDispatcher` (routing a typed line to a reply) is pure logic
and unit-tested; :func:`run_terminal_chat` is the thin readline loop on top.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

from lib.telegram.userinfo import parse_command


LOGGER = logging.getLogger("lib.console")

# A console "chat id" — the providers expect one; the value is irrelevant since
# replies are routed to the terminal, not a Telegram chat.
CONSOLE_CHAT_ID = 0

HELP_TEXT = (
    "Talk to the aviary — commands or plain language:\n"
    "  /activity [bird] [today]   activity summary (e.g. /activity percy today)\n"
    "  /status            camera + detection status boxes\n"
    "  /find <bird>       e.g. /find percy, /find the cockatiels, /find stop\n"
    "  /stop [time]       privacy mode (/stop 10m); /start to resume\n"
    "  /discover          rescan the network for cameras\n"
    "  /home              aim pan-tilt cameras at their saved viewpoint\n"
    "  /snapshot          capture all cameras (photos go to Telegram)\n"
    "  /logs              show/hide live server logs\n"
    "  /help   /quit\n"
    "  …or just type: \"where's percy?\", \"what did matcha do today?\", \"stop the cams\""
)


class ConsoleNotifier:
    """A notifier shim that routes messages to a console ``emit`` instead of Telegram.

    Quacks like :class:`~lib.telegram.notifier.TelegramNotifier` for the bits the
    finder / NL router use (``send_text`` / ``send_album``). Photos are NOT
    printed — the terminal stays text-only — so an album becomes a short note.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self.user_ids = ["console"]
        self.bot_token = "console"

    def send_text(self, chat_id, text: str) -> None:
        self._emit(text)

    def send_album(self, chat_id, items) -> None:
        count = len(list(items))
        if count:
            self._emit(f"[sent {count} photo(s) to Telegram]")

    def broadcast_text(self, text: str) -> None:
        self._emit(text)

    def broadcast_album(self, items) -> None:
        self.send_album(None, items)

    def send_chat_action(self, chat_id, action: str = "typing") -> None:
        # No typing indicator in a terminal — the reply just prints when ready.
        return None


class ConsoleDispatcher:
    """Routes a typed line to the right provider and returns/echoes a reply."""

    def __init__(
        self,
        *,
        emit: Callable[[str], None],
        status_text: Callable[[], str],
        discover_text: Callable[[], str],
        snapshot_text: Callable[[int], str],
        pause: Callable[[float | None], str],
        resume: Callable[[], str],
        find: Callable[[int, str], str],
        nl_handle: Callable[[int, str], None],
        parse_duration: Callable[[str | None], float | None],
        activity: Callable[[int, str], None] | None = None,
        home_text: Callable[[], str] | None = None,
        toggle_logs: Callable[[], str] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self._emit = emit
        self._status_text = status_text
        self._discover_text = discover_text
        self._snapshot_text = snapshot_text
        self._pause = pause
        self._resume = resume
        self._find = find
        self._nl_handle = nl_handle
        self._parse_duration = parse_duration
        self._activity = activity
        self._home_text = home_text
        self._toggle_logs = toggle_logs
        self._on_quit = on_quit

    def handle(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        command = parse_command(line)
        if command is None:
            # Plain language -> the natural-language router (async; prints later).
            self._nl_handle(CONSOLE_CHAT_ID, line)
            return

        argument = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else ""
        if command in ("/quit", "/exit", "/q"):
            self._emit("Bye! 👋")
            if self._on_quit:
                self._on_quit()
        elif command == "/help":
            self._emit(HELP_TEXT)
        elif command in ("/logs",):
            self._emit(self._toggle_logs() if self._toggle_logs else "Logs toggle unavailable.")
        elif command == "/activity":
            if self._activity is not None:
                self._emit("📋 Looking back…")
                self._activity(CONSOLE_CHAT_ID, argument)
            else:
                self._emit("Activity memory is off (needs Ollama).")
        elif command in ("/status", "/cams"):
            self._emit(self._status_text())
        elif command in ("/pause", "/stop"):
            self._emit(self._pause(self._parse_duration(argument)))
        elif command in ("/play", "/resume", "/start"):
            self._emit(self._resume())
        elif command == "/find":
            self._emit(self._find(CONSOLE_CHAT_ID, argument))
        elif command == "/discover":
            self._emit("Scanning the local network for cameras…")
            self._emit(self._discover_text())
        elif command == "/home":
            self._emit(self._home_text() if self._home_text is not None else "PTZ control is off.")
        elif command == "/snapshot":
            self._emit("Capturing snapshots from all cameras…")
            self._emit(self._snapshot_text(CONSOLE_CHAT_ID))
        else:
            self._emit(f"Unknown command {command}. Type /help.")


class ConsoleLogToggle:
    """Attach/detach a stderr log handler on demand (the /logs menu)."""

    def __init__(self, level: int = logging.INFO) -> None:
        self._handler: logging.Handler | None = None
        self._level = level

    def toggle(self) -> str:
        root = logging.getLogger()
        if self._handler is None:
            # stdout on purpose: in chat mode native stderr is redirected to the
            # logfile, so a stderr handler would never reach the terminal.
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self._level)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            root.addHandler(handler)
            self._handler = handler
            return "Live logs ON (type /logs to hide)."
        root.removeHandler(self._handler)
        self._handler = None
        return "Live logs OFF."


def run_terminal_chat(dispatcher: ConsoleDispatcher, stop_event) -> None:
    """Read lines from stdin and dispatch them until EOF / quit / stop.

    Reads on a daemon thread and polls a queue so the loop also wakes on
    ``stop_event`` — that way an external SIGTERM (server shutdown) stops the
    chat too, not just Ctrl-C/Ctrl-D. Deliberately a simple readline UI (no
    full-screen takeover) so it composes with normal terminal scrollback.
    """
    import queue
    import threading

    print("\n🐦 Aviary console — type /help, or just chat. Ctrl-D to exit.\n")
    lines: "queue.Queue[str | None]" = queue.Queue()

    def reader() -> None:
        while True:
            try:
                lines.put(input("aviary> "))
            except (EOFError, KeyboardInterrupt):
                lines.put(None)
                return

    threading.Thread(target=reader, name="console-input", daemon=True).start()

    while not stop_event.is_set():
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        if line is None:  # Ctrl-D / EOF
            print()
            break
        try:
            dispatcher.handle(line)
        except Exception:
            LOGGER.exception("Console command failed")
            print("Something went wrong handling that.")
    stop_event.set()
