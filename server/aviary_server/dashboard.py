"""Live terminal dashboard for the camera monitor.

Renders one in-place cell per camera (status + rolling statistics) plus a
scrolling events panel, so an operator can see at a glance which streams are
healthy and why one isn't. The capture threads only mutate their own
``CameraStats``; the dashboard thread reads snapshots and repaints.

Two things make the display usable in practice:

* OpenCV/FFmpeg write h264 decode errors and RTSP timeout warnings straight to
  file descriptor 2, bypassing Python logging. Left alone they shred any
  in-place render. On a real terminal we redirect fd 2 to a logfile (kept for
  deep debugging) and silence OpenCV's own logger, so the screen stays clean.
* Under ``docker compose`` stdout is not an in-place-capable TTY. There we skip
  the Live render entirely and emit a compact status line on an interval, which
  reads fine in ``docker logs``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

try:  # OpenCV is always present (transitive via ultralytics); guard anyway.
    import cv2
except Exception:  # pragma: no cover - cv2 import is exercised everywhere else
    cv2 = None

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


LOGGER = logging.getLogger("aviary_server")

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


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


class ObjectRegistry:
    """Global tally of every label seen across all cameras.

    Shared by every :class:`CameraStats`; the dashboard reads it to render the
    "objects" section. Each label tracks when it was last seen, how many frames
    it has appeared in, and which cameras saw it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, dict] = {}

    def record(self, labels: list[str], camera_name: str) -> None:
        now = time.monotonic()
        with self._lock:
            for label in labels:
                entry = self._objects.get(label)
                if entry is None:
                    entry = {"count": 0, "cameras": set()}
                    self._objects[label] = entry
                entry["last_seen"] = now
                entry["count"] += 1
                entry["cameras"].add(camera_name)

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            rows = [
                {
                    "label": label,
                    "since": now - entry["last_seen"],
                    "count": entry["count"],
                    "cameras": sorted(entry["cameras"]),
                }
                for label, entry in self._objects.items()
            ]
        # Smallest "since" (most recently seen) first.
        rows.sort(key=lambda row: row["since"])
        return rows


class CameraStats:
    """Thread-safe rolling metrics for a single camera.

    The owning capture thread calls the ``record_*``/``set_status`` mutators;
    the dashboard thread calls :meth:`snapshot`. All access is guarded so a
    repaint never sees a half-updated set of counters.
    """

    def __init__(
        self,
        name: str,
        sample_fps: float,
        registry: ObjectRegistry | None = None,
        fps_window: float = 2.0,
    ) -> None:
        self.name = name
        self.sample_fps = sample_fps
        self._registry = registry
        self._fps_window = fps_window
        self._lock = threading.Lock()

        self.status = "connecting"
        self.backoff = 0.0
        self.frames_total = 0
        self.detections_total = 0
        self.alerts_sent = 0
        self.reconnects = 0
        self.consecutive_failures = 0
        self.fps = 0.0
        self.last_label: str | None = None
        self.last_detection_at: float | None = None
        self.last_frame_at: float | None = None
        self.started_at = time.monotonic()

        self._fps_count = 0
        self._fps_window_start = time.monotonic()

    def set_status(self, status: str, backoff: float = 0.0) -> None:
        with self._lock:
            self.status = status
            self.backoff = backoff

    def record_inference(self, labels: list[str]) -> None:
        now = time.monotonic()
        with self._lock:
            self.frames_total += 1
            self.last_frame_at = now
            self.consecutive_failures = 0
            self._fps_count += 1
            elapsed = now - self._fps_window_start
            if elapsed >= self._fps_window:
                self.fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_window_start = now
            if labels:
                self.detections_total += len(labels)
                self.last_label = ", ".join(sorted(set(labels)))
                self.last_detection_at = now
        # Update the shared registry outside this camera's lock to avoid holding
        # two locks at once.
        if labels and self._registry is not None:
            self._registry.record(sorted(set(labels)), self.name)

    def record_read_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1

    def record_reconnect(self) -> None:
        # A torn-down session: zero the live rate so a stalled cell doesn't keep
        # showing a stale FPS from before the drop.
        with self._lock:
            self.reconnects += 1
            self.fps = 0.0
            self._fps_count = 0
            self._fps_window_start = time.monotonic()

    def record_alert(self, count: int = 1) -> None:
        with self._lock:
            self.alerts_sent += count

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            since_frame = None if self.last_frame_at is None else now - self.last_frame_at
            since_detection = (
                None if self.last_detection_at is None else now - self.last_detection_at
            )
            return {
                "name": self.name,
                "sample_fps": self.sample_fps,
                "status": self.status,
                "backoff": self.backoff,
                "frames_total": self.frames_total,
                "detections_total": self.detections_total,
                "alerts_sent": self.alerts_sent,
                "reconnects": self.reconnects,
                "consecutive_failures": self.consecutive_failures,
                "fps": self.fps,
                "last_label": self.last_label,
                "since_detection": since_detection,
                "since_frame": since_frame,
                "uptime": now - self.started_at,
            }


class _EventLogHandler(logging.Handler):
    """Feeds Python log records into the dashboard's events panel."""

    def __init__(self, dashboard: "Dashboard") -> None:
        super().__init__()
        self._dashboard = dashboard

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._dashboard.add_event(record.getMessage(), record.levelname)
        except Exception:  # never let logging crash the app
            pass


class Dashboard:
    """Owns the render loop, the events buffer, and log/stderr plumbing."""

    def __init__(
        self,
        stats: dict[str, CameraStats],
        registry: ObjectRegistry | None = None,
        log_level: int = logging.INFO,
        logfile: str = "aviary.log",
        refresh_per_second: int = 4,
        status_line_interval: float = 10.0,
    ) -> None:
        self.console = Console()
        self.is_tty = self.console.is_terminal
        self.stats = stats
        self.registry = registry
        self._log_level = log_level
        self._logfile_path = logfile
        self._refresh = refresh_per_second
        self._status_line_interval = status_line_interval

        self._events: deque[tuple[str, str, str]] = deque(maxlen=8)
        self._events_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved_stderr_fd: int | None = None
        self._logfile = None

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
        self._install_logging()
        if self.is_tty:
            self._silence_native_stderr()
        self._thread = threading.Thread(target=self._run, name="dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._restore_native_stderr()

    # -- logging / native-noise plumbing --------------------------------

    def _install_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(self._log_level)
        for handler in list(root.handlers):
            root.removeHandler(handler)

        # Always tee everything to the logfile so the h264/timeout detail and
        # full tracebacks survive for post-mortem, regardless of render mode.
        file_handler = logging.FileHandler(self._logfile_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(file_handler)

        if self.is_tty:
            # On a terminal the events panel is the on-screen log; a stream
            # handler would scribble over the Live render.
            root.addHandler(_EventLogHandler(self))
        else:
            # Non-TTY (docker logs): plain stream logging is the right thing.
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            root.addHandler(stream_handler)

    def _silence_native_stderr(self) -> None:
        if cv2 is not None:
            try:
                cv2.setLogLevel(0)  # 0 == LOG_LEVEL_SILENT
            except Exception:
                pass
        try:
            self._logfile = open(self._logfile_path, "a", buffering=1)
            self._saved_stderr_fd = os.dup(2)
            os.dup2(self._logfile.fileno(), 2)
        except Exception:
            self._saved_stderr_fd = None

    def _restore_native_stderr(self) -> None:
        if self._saved_stderr_fd is not None:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
            self._saved_stderr_fd = None
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None

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
        # screen is restored on exit, so the final frame isn't left behind —
        # the logfile remains the post-mortem record.
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
                f"alerts={snap['alerts_sent']} fails={snap['consecutive_failures']}"
            )
        LOGGER.info("status | %s", " | ".join(parts))

    # -- rendering -------------------------------------------------------

    def _render(self) -> Layout:
        """Compose a full-terminal layout: camera cells, objects, events.

        Rebuilt every tick from the live console size, so it tracks terminal
        resizes. The camera band is sized to exactly fit its grid of cells, the
        events band is fixed, and the objects band flexes to absorb the rest of
        the height. Camera cells expand across the full width.
        """
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
            body = Table.grid(padding=(0, 2))
            body.add_column(style="cyan")  # object
            body.add_column(justify="right")  # last seen
            body.add_column(justify="right", style="grey62")  # sightings
            body.add_column(style="grey50")  # cameras
            for row in rows:
                since = row["since"]
                colour = "green" if since < 5 else "yellow" if since < 30 else "grey50"
                body.add_row(
                    Text(row["label"], style="bold cyan"),
                    Text(f"{_format_duration(since)} ago", style=colour),
                    _format_count(row["count"]),
                    ", ".join(row["cameras"]),
                )
        return Panel(body, title="objects (most recent first)", border_style="grey37")

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
