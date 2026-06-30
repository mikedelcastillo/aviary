"""Live terminal dashboard for the camera monitor."""

from __future__ import annotations

import logging
import math
import threading
from collections import deque

from lib.clock import now_ph

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lib.discovery import HOST_FAILED, HOST_FOUND, HOST_PENDING, HOST_TESTING
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
    "paused": ("⏸", "magenta"),
}

# How long a "connected" camera may go without a decoded frame before the cell
# flags the stream as stalled (it's nominally up but delivering nothing).
STALE_FRAME_SECONDS = 8.0

# --- Content-aware camera-band geometry ------------------------------------
# The camera band must adapt to the terminal: when there's room we show full
# per-camera detail panels in a wrapping grid; when there isn't, we fall back to
# a dense one-line-per-camera view so EVERY camera's status stays visible. These
# constants describe the real footprint of each presentation so the mode/column
# math (which is pure and unit-tested) lines up with what Rich actually renders.

# A detail panel's table is 9 rows; the panel border adds 2 -> 11 lines tall.
DETAIL_PANEL_HEIGHT = 11
# Minimum readable width for a detail cell. Cells narrower than this wrap their
# values badly, so we cap the column count to keep each cell at least this wide.
DETAIL_MIN_CELL_WIDTH = 28
# Minimum width for a single compact one-liner column. Below this the status
# text truncates uncomfortably, so we pack fewer columns instead.
COMPACT_MIN_CELL_WIDTH = 26

# --- Discovery debug grid --------------------------------------------------
# While a discovery sweep is running, the camera band is replaced by a grid with
# one cell per scanned IP, coloured by the host's live probe state. A cell shows
# the host's last octet right-justified to 3 chars plus a trailing space.
DISCOVERY_CELL_WIDTH = 4
DISCOVERY_STATE_STYLE = {
    HOST_PENDING: "grey42",      # queued, not yet probed
    HOST_TESTING: "yellow",      # probe in flight
    HOST_FOUND: "bold green",    # confirmed camera
    HOST_FAILED: "red",          # port closed / auth / no stream
}


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


def _short_name(snap: dict) -> str:
    """Best display label for the compact view.

    The supervisor names cameras ``camera-<host>`` so the IP is already embedded
    in the name; if a ``host`` field is ever added to the snapshot we prefer it.
    Either way we strip the ``camera-`` prefix so the dense view stays narrow.
    """
    host = snap.get("host")
    if host:
        return str(host)
    name = str(snap.get("name", ""))
    if name.startswith("camera-"):
        return name[len("camera-"):]
    return name


def _camera_layout_mode(width: int, band_height: int, n: int) -> str:
    """Decide DETAILED vs COMPACT for ``n`` cameras at a given band size.

    Pure function of (width, band_height, n) so it is unit-testable without a
    real terminal. We prefer DETAILED whenever the wrapping grid of full panels
    fits the band; otherwise we fall back to COMPACT, which can always show every
    camera. With zero cameras the placeholder is rendered, so the mode is moot —
    we report COMPACT (cheapest path) for determinism.
    """
    if n <= 0:
        return "compact"
    columns = _detail_columns(width, n)
    rows = math.ceil(n / columns)
    if rows * DETAIL_PANEL_HEIGHT <= band_height:
        return "detailed"
    return "compact"


def _discovery_columns(width: int, n: int) -> int:
    """Column count for the discovery grid: as many fixed-width cells as fit.

    Pure function of (width, n) so the grid geometry is unit-testable without a
    terminal. Uses the full width (fixed ``DISCOVERY_CELL_WIDTH`` cells) to
    minimise the number of rows, so a whole /24 fits in as little height as
    possible. Never more columns than hosts.
    """
    if n <= 0:
        return 1
    return max(1, min(n, width // DISCOVERY_CELL_WIDTH))


def _detail_columns(width: int, n: int) -> int:
    """Column count for the DETAILED wrapping grid.

    Clamp ``width // DETAIL_MIN_CELL_WIDTH`` into ``[2, n]`` so cells never get
    narrower than is readable, but allow a single column only when there is a
    single camera (the contract's "never show fewer than 2" invariant means we
    keep at least 2 columns whenever ``n >= 2``).
    """
    if n <= 1:
        return 1
    by_width = max(1, width // DETAIL_MIN_CELL_WIDTH)
    return max(2, min(by_width, n))


def _compact_columns(width: int, n: int) -> int:
    """Column count for the COMPACT multi-column packing.

    Each column needs at least ``COMPACT_MIN_CELL_WIDTH``; we never use more
    columns than there are cameras. At least one column is always returned.
    """
    if n <= 0:
        return 1
    by_width = max(1, width // COMPACT_MIN_CELL_WIDTH)
    return max(1, min(by_width, n))


def _compact_rows(n: int, columns: int) -> int:
    """Rows needed to lay ``n`` one-liners across ``columns`` columns."""
    columns = max(1, columns)
    return math.ceil(n / columns) if n else 1


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
        stats_lock: threading.Lock | None = None,
        discovery_progress=None,
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
        # The supervisor mutates the shared stats dict from another thread (it
        # registers a new CameraStats whenever /discover finds a camera). Every
        # iteration over self.stats must snapshot under this lock, otherwise the
        # render thread can hit "dict changed size during iteration". If the
        # caller doesn't supply one we still create a real lock so the snapshot
        # helper is always safe to call.
        self._stats_lock = stats_lock if stats_lock is not None else threading.Lock()
        # Live discovery sweep state (a lib.discovery.DiscoveryProgress, or any
        # object exposing snapshot()/is_active()). When a sweep is active the
        # camera band is replaced by the colour-coded discovery grid. Duck-typed
        # and optional so the dashboard works with or without one.
        self._discovery_progress = discovery_progress

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
        timestamp = now_ph().strftime("%H:%M:%S")
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

    # -- thread-safe stats access ----------------------------------------

    def _camera_snapshots(self) -> list[dict]:
        """Snapshot every camera's stats under the lock.

        Snapshots are taken from a list() copy of the values while holding the
        lock so the supervisor can keep adding cameras concurrently without ever
        tripping a "dict changed size during iteration" error. The returned list
        of plain dicts is then safe to iterate freely off-lock.
        """
        with self._stats_lock:
            cameras = list(self.stats.values())
        return [camera.snapshot() for camera in cameras]

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
        # Snapshot under the lock; never iterate the live dict directly.
        for snap in self._camera_snapshots():
            parts.append(
                f"{snap['name']}={snap['status'].upper()} "
                f"fps={snap['fps']:.1f} frames={snap['frames_total']} "
                f"last_frame={_format_frame_age(snap['since_frame'])} "
                f"alerts={snap['alerts_sent']} fails={snap['consecutive_failures']}"
            )
        LOGGER.info("status | %s", " | ".join(parts))

    # -- rendering -------------------------------------------------------

    def _render(self) -> Layout:
        """Compose a full-terminal layout: camera band, objects, events."""
        width, height = self.console.size
        snaps = self._camera_snapshots()

        events_height = self._events.maxlen + 2  # rows + panel border
        objects_min = 3
        # The camera band may grow but must always leave room for the objects and
        # events panels below it (same cap discipline as before).
        max_band = max(objects_min, height - events_height - objects_min)

        # While a discovery sweep is running, the camera band becomes the live
        # colour-coded grid of every IP being probed. It reverts the moment the
        # sweep ends (active flips off) and the freshly-found cameras appear.
        progress = None
        if self._discovery_progress is not None:
            snap = self._discovery_progress.snapshot()
            if snap.get("active"):
                progress = snap
        if progress is not None:
            cameras, camera_height = self._discovery_band(progress, width, max_band)
        else:
            cameras, camera_height = self._camera_band(snaps, width, max_band)

        layout = Layout()
        layout.split_column(
            Layout(cameras, name="cameras", size=camera_height),
            Layout(self._objects_panel(), name="objects", ratio=1, minimum_size=objects_min),
            Layout(self._events_panel(), name="events", size=events_height),
        )
        return layout

    def _camera_band(self, snaps: list[dict], width: int, max_band: int):
        """Build the content-aware camera band and report its height.

        Returns ``(renderable, height)``. The height is the band's actual line
        count clamped into ``[1, max_band]`` so the objects/events panels keep
        their room. The mode is chosen by the pure helpers above so the geometry
        is testable without a terminal.
        """
        n = len(snaps)
        if n == 0:
            # Friendly nudge: nothing is running yet, point the user at the bot.
            placeholder = Panel(
                Text("no cameras yet — send /discover", style="grey50", justify="center"),
                title="cameras",
                border_style="grey37",
            )
            return placeholder, min(3, max_band)

        mode = _camera_layout_mode(width, max_band, n)
        if mode == "detailed":
            return self._detailed_band(snaps, width, max_band)
        return self._compact_band(snaps, width, max_band)

    def _detailed_band(self, snaps: list[dict], width: int, max_band: int):
        """Full per-camera panels arranged in a wrapping grid."""
        n = len(snaps)
        columns = _detail_columns(width, n)
        rows = math.ceil(n / columns)

        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in range(columns):
            grid.add_column(ratio=1)
        panels = [self._camera_panel(snap) for snap in snaps]
        for r in range(rows):
            cells = panels[r * columns:(r + 1) * columns]
            # Pad the final short row so every column keeps an equal width.
            cells += [Text("")] * (columns - len(cells))
            grid.add_row(*cells)

        # Each grid row is one panel tall; clamp so objects/events keep room.
        band_height = min(rows * DETAIL_PANEL_HEIGHT, max_band)
        band_height = max(1, band_height)
        return grid, band_height

    def _compact_band(self, snaps: list[dict], width: int, max_band: int):
        """Dense one-line-per-camera view that shows EVERY camera's status.

        Packs the one-liners into multiple width-sized columns and as many rows
        as the band allows; columns/rows are chosen so all ``n`` statuses fit (we
        never drop a camera in compact mode). A healthy/total count goes in the
        title so it's obvious this is the dense fallback view.
        """
        n = len(snaps)
        columns = _compact_columns(width, n)
        rows = _compact_rows(n, columns)

        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in range(columns):
            grid.add_column(ratio=1)

        lines = [self._compact_line(snap) for snap in snaps]
        for r in range(rows):
            cells = lines[r * columns:(r + 1) * columns]
            cells += [Text("")] * (columns - len(cells))
            grid.add_row(*cells)

        healthy = sum(1 for snap in snaps if self._is_healthy(snap))
        title = f"cameras (compact) — {healthy}/{n} healthy"
        panel = Panel(grid, title=title, border_style="grey37")
        # Panel adds a 2-line border on top of the packed rows; clamp to the band.
        band_height = min(rows + 2, max_band)
        band_height = max(1, band_height)
        return panel, band_height

    def _discovery_band(self, progress: dict, width: int, max_band: int):
        """Live grid of every scanned IP, coloured by its probe state.

        Replaces the camera band while a sweep runs. One cell per host (last
        octet, right-justified) coloured grey/yellow/green/red for
        pending/testing/found/failed. A legend + running counts sit above the
        grid so the colours are self-explanatory. Returns ``(renderable,
        height)`` clamped to the band budget like the other bands.
        """
        order = progress.get("order", [])
        states = progress.get("states", {})
        counts = progress.get("counts", {})
        network = progress.get("network", "")
        n = len(order)

        columns = _discovery_columns(width, n)
        rows = math.ceil(n / columns) if n else 1

        grid = Table.grid(expand=True, padding=0)
        for _ in range(columns):
            grid.add_column(justify="right", no_wrap=True)

        cells = [
            Text(
                f"{host.rsplit('.', 1)[-1]:>3} ",
                style=DISCOVERY_STATE_STYLE.get(states.get(host), "grey42"),
            )
            for host in order
        ]
        for r in range(rows):
            row = cells[r * columns:(r + 1) * columns]
            row += [Text("")] * (columns - len(row))
            grid.add_row(*row)

        legend = Text.assemble(
            ("● pending  ", DISCOVERY_STATE_STYLE[HOST_PENDING]),
            ("● testing  ", DISCOVERY_STATE_STYLE[HOST_TESTING]),
            ("● found  ", DISCOVERY_STATE_STYLE[HOST_FOUND]),
            ("● none", DISCOVERY_STATE_STYLE[HOST_FAILED]),
        )
        scope = f"{network}0/24" if network else "network"
        title = (
            f"discovery — scanning {scope}  ·  "
            f"{counts.get(HOST_FOUND, 0)} found / "
            f"{counts.get(HOST_TESTING, 0)} testing / "
            f"{counts.get(HOST_FAILED, 0)} done / "
            f"{counts.get(HOST_PENDING, 0)} pending of {n}"
        )
        panel = Panel(Group(legend, grid), title=title, border_style="cyan")
        # Border (2) + legend (1) on top of the packed grid rows; clamp to band.
        band_height = max(1, min(rows + 3, max_band))
        return panel, band_height

    def _is_healthy(self, snap: dict) -> bool:
        """A camera counts as healthy when connected and delivering frames."""
        if snap["status"] != "connected":
            return False
        since_frame = snap["since_frame"]
        return since_frame is None or since_frame <= STALE_FRAME_SECONDS

    def _compact_line(self, snap: dict) -> Text:
        """One colour-coded status line for a single camera.

        Reuses the same glyph/colour mapping, stall detection and fps/frame
        colouring as the detail cells, so the dense view stays consistent with
        the panels: glyph, shortened name, STATUS, fps vs target, frame age,
        alerts and fails.
        """
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

        line = Text.assemble((f"{glyph} ", colour), (_short_name(snap), "bold"))
        line.append("  ")
        line.append(Text(status_text, style=f"bold {colour}"))
        line.append(" · ")
        line.append(self._fps_text(snap, connected))
        line.append(" · ")
        line.append(self._frame_text(snap))
        line.append(Text(f"  a{snap['alerts_sent']}", style="grey62"))
        fails = snap["consecutive_failures"]
        line.append(Text(f" f{fails}", style="red" if fails else "grey62"))
        return line

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
