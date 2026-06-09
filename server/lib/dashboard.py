"""Live terminal dashboard for the camera monitor."""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lib.objects import ObjectRegistry
from lib.stats import CameraStats
from lib.terminal_logging import NativeStderrRedirect, configure_dashboard_logging


LOGGER = logging.getLogger("lib.dashboard")

# Glyph + colour per lifecycle state. Keys are the only valid status strings.
STATUS_STYLE = {
    "connecting": ("◐", "yellow"),
    "connected": ("●", "green"),
    "reconnecting": ("○", "red"),
    "stopped": ("■", "grey50"),
}

# How long a "connected" camera may go without a decoded frame before the cell
# flags the stream as stalled (it's nominally up but delivering nothing).
STALE_FRAME_SECONDS = 8.0


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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


class Dashboard:
    """Owns the render loop and dashboard event buffer."""

    def __init__(
        self,
        stats: dict[str, CameraStats],
        registry: ObjectRegistry | None = None,
        log_level: int = logging.INFO,
        logfile: str = "aviary.log",
        refresh_per_second: int = 4,
        status_line_interval: float = 10.0,
        movement_alert_ratio: float = 0.10,
    ) -> None:
        self.console = Console()
        self.is_tty = self.console.is_terminal
        self.stats = stats
        self.registry = registry
        self._log_level = log_level
        self._logfile_path = logfile
        self._refresh = refresh_per_second
        self._status_line_interval = status_line_interval
        self._movement_alert_ratio = movement_alert_ratio

        self._events: deque[tuple[str, str, str]] = deque(maxlen=8)
        self._events_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stderr_redirect = NativeStderrRedirect(logfile)

    # -- public API ------------------------------------------------------

    def record_alert(self, camera_name: str, count: int = 1) -> None:
        stats = self.stats.get(camera_name)
        if stats is not None:
            stats.record_alert(count)

    def add_event(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._events_lock:
            self._events.append((timestamp, level.upper(), message))

    def start(self) -> None:
        configure_dashboard_logging(
            add_event=self.add_event,
            is_tty=self.is_tty,
            log_level=self._log_level,
            logfile=self._logfile_path,
        )
        if self.is_tty:
            self._stderr_redirect.start()
        self._thread = threading.Thread(target=self._run, name="dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._stderr_redirect.stop()

    # -- render loop -----------------------------------------------------

    def _run(self) -> None:
        if self.is_tty:
            self._run_live()
        else:
            self._run_plain()

    def _run_live(self) -> None:
        from rich.live import Live

        # screen=True takes the full terminal (alternate buffer), so the layout
        # fills the entire width and height and tracks resizes. Trade-off: the
        # screen is restored on exit, so the final frame isn't left behind; the
        # logfile remains the post-mortem record.
        with Live(
            self._render(),
            console=self.console,
            refresh_per_second=self._refresh,
            screen=True,
        ) as live:
            while not self._stop.is_set():
                live.update(self._render())
                self._stop.wait(1.0 / self._refresh)
            live.update(self._render())

    def _run_plain(self) -> None:
        while not self._stop.is_set():
            self._emit_status_line()
            self._stop.wait(self._status_line_interval)

    def _emit_status_line(self) -> None:
        parts = []
        for name in self.stats:
            snap = self.stats[name].snapshot()
            parts.append(
                f"{name}={snap['status'].upper()} "
                f"fps={snap['fps']:.1f} frames={snap['frames_total']} "
                f"last_frame={_format_frame_age(snap['since_frame'])} "
                f"alerts={snap['alerts_sent']} fails={snap['consecutive_failures']}"
            )
        LOGGER.info("status | %s", " | ".join(parts))

    # -- rendering -------------------------------------------------------

    def _render(self) -> Layout:
        """Compose a full-terminal layout: camera cells, objects, events."""
        width, height = self.console.size
        panels = [self._camera_panel(self.stats[name].snapshot()) for name in self.stats]
        # One row, equal-width columns that fill the terminal width.
        cameras = Table.grid(expand=True, padding=(0, 1))
        for _ in panels:
            cameras.add_column(ratio=1)
        cameras.add_row(*panels)

        # Measure the band's true height at this width so narrow cells that wrap
        # a value (e.g. "RECONNECTING (20s)") aren't clipped.
        camera_options = self.console.options.update(width=width, height=None)
        camera_height = max(1, len(self.console.render_lines(cameras, camera_options, pad=False)))
        events_height = self._events.maxlen + 2  # rows + panel border
        objects_min = 3
        camera_height = min(camera_height, max(objects_min, height - events_height - objects_min))

        layout = Layout()
        layout.split_column(
            Layout(cameras, name="cameras", size=camera_height),
            Layout(self._objects_panel(), name="objects", ratio=1, minimum_size=objects_min),
            Layout(self._events_panel(), name="events", size=events_height),
        )
        return layout

    def _camera_panel(self, snap: dict) -> Panel:
        glyph, colour = STATUS_STYLE.get(snap["status"], ("?", "white"))
        connected = snap["status"] == "connected"
        stalled = (
            connected
            and snap["since_frame"] is not None
            and snap["since_frame"] > STALE_FRAME_SECONDS
        )

        status_text = snap["status"].upper()
        if snap["status"] == "reconnecting" and snap["backoff"]:
            status_text += f" ({snap['backoff']:.0f}s)"
        if stalled:
            status_text = "STALLED"
            colour = "red"

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="grey62")
        table.add_column()

        table.add_row("status", Text(status_text, style=f"bold {colour}"))
        table.add_row("fps", self._fps_text(snap, connected))
        table.add_row("frame", self._frame_text(snap))
        table.add_row("frames", _format_count(snap["frames_total"]))
        table.add_row("detect", self._detection_text(snap))
        table.add_row("alerts", str(snap["alerts_sent"]))
        table.add_row("fails", self._failures_text(snap))
        table.add_row("reconns", str(snap["reconnects"]))
        table.add_row("uptime", _format_duration(snap["uptime"]))

        title = Text.assemble((f"{glyph} ", colour), (snap["name"], "bold"))
        # No fixed width: the cell stretches to fill its equal-ratio column.
        return Panel(table, title=title, border_style=colour)

    def _fps_text(self, snap: dict, connected: bool) -> Text:
        fps = snap["fps"]
        target = snap["sample_fps"] or 1.0
        if not connected:
            return Text(f"{fps:.2f}", style="grey50")
        ratio = fps / target
        colour = "green" if ratio >= 0.8 else "yellow" if ratio >= 0.4 else "red"
        return Text(f"{fps:.2f}", style=colour) + Text(f" / {target:g}", style="grey50")

    def _frame_text(self, snap: dict) -> Text:
        since_frame = snap["since_frame"]
        if since_frame is None:
            return Text("never", style="grey50")
        if since_frame > STALE_FRAME_SECONDS:
            style = "red"
        elif since_frame >= 1.0:
            style = "yellow"
        else:
            style = "green"
        return Text(f"{_format_frame_age(since_frame)} ago", style=style)

    def _detection_text(self, snap: dict) -> Text:
        if not snap["last_label"]:
            return Text("—", style="grey50")
        age = "" if snap["since_detection"] is None else f" ({_format_duration(snap['since_detection'])} ago)"
        return Text(snap["last_label"], style="cyan") + Text(age, style="grey50")

    def _failures_text(self, snap: dict) -> Text:
        fails = snap["consecutive_failures"]
        return Text(str(fails), style="red" if fails else "grey62")

    def _objects_panel(self) -> Panel:
        rows = self.registry.snapshot() if self.registry is not None else []
        if not rows:
            body: Text | Table = Text("(nothing seen yet)", style="grey50")
        else:
            body = Table(expand=False, show_edge=False, box=None)
            body.add_column("Camera", style="grey62")
            body.add_column("Object", style="cyan")
            body.add_column("Last Seen", style="grey62")
            body.add_column("Last Alert", style="grey62")
            body.add_column("Move")
            body.add_column("Seen", style="grey62")
            for row in rows:
                since = row["since"]
                colour = "green" if since < 5 else "yellow" if since < 30 else "grey50"
                since_alert = row["since_alert"]
                if since_alert is None:
                    alert = Text("—", style="grey50")
                else:
                    alert = Text(f"{_format_duration(since_alert)} ago", style="grey62")
                movement_percent = row["movement_percent"]
                if movement_percent is None:
                    movement = Text("—", style="grey50")
                else:
                    threshold_percent = self._movement_alert_ratio * 100
                    movement_colour = "yellow" if movement_percent >= threshold_percent else "grey62"
                    movement = Text(f"{movement_percent:.1f}%", style=movement_colour)
                body.add_row(
                    row["camera"],
                    Text(row["label"], style="bold cyan"),
                    Text(f"{_format_duration(since)} ago", style=colour),
                    alert,
                    movement,
                    _format_count(row["count"]),
                )
        return Panel(body, title="objects by camera (most recent first)", border_style="grey37")

    def _events_panel(self) -> Panel:
        with self._events_lock:
            events = list(self._events)
        if not events:
            body: Text | Table = Text("(no events yet)", style="grey50")
        else:
            body = Table.grid(padding=(0, 1))
            body.add_column(style="grey50")
            body.add_column()
            for timestamp, level, message in events:
                level_colour = {
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red",
                }.get(level, "grey62")
                body.add_row(
                    Text(timestamp, style="grey50"),
                    Text(message, style=level_colour),
                )
        return Panel(body, title="events", border_style="grey37")
