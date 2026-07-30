"""RGBController — the background engine that turns aviary state into light.

Mirrors the codebase's shared-state pattern (a daemon thread + a lock, like
MachineSampler/IRState): state setters are called from other threads
(discovery callback, camera loop, IR listener), a render loop runs at a fixed
FPS picking the highest-priority scene and pushing it to the LED surface.

Priority (high → low):  MANUAL ▸ BOOT ▸ ERROR ▸ DISCOVERY ▸ NIGHT ▸ DETECTION ▸ IDLE

Everything degrades gracefully: if the OpenRGB server is down the surface is a
no-op and the loop keeps trying to reconnect, so the aviary server never depends
on the lights being present.
"""
from __future__ import annotations

import logging
import re
import threading
import time

from . import birds, scenes
from .layout import Layout, load_layout
from .palette import BLACK, Color
from .scenes import SceneState, Sighting
from .surface import OpenRGBSurface, Surface, SurfaceConfig

log = logging.getLogger("aviary.rgb.engine")

BOOT_SECONDS = 4.0
DEFAULT_MANUAL_SECONDS = 600.0  # /rgb red  -> 10 minutes, then back to auto
# Reconnect probing backs off exponentially from the base to the cap while the
# OpenRGB server stays away, so a box that simply has no LED rig doesn't retry
# (and log) every 5s forever — that alone once wrote 200k+ log lines.
RECONNECT_EVERY = 5.0
RECONNECT_MAX = 600.0

# Named colors accepted by /rgb (plus any bird name and #hex / hex).
NAMED_COLORS: dict[str, Color] = {
    "red": Color(255, 0, 0), "green": Color(0, 255, 0), "blue": Color(0, 0, 255),
    "white": Color(255, 255, 255), "black": Color(0, 0, 0), "off": Color(0, 0, 0),
    "yellow": Color(255, 200, 0), "gold": Color(255, 208, 0), "amber": Color(255, 150, 0),
    "orange": Color(255, 110, 0), "purple": Color(150, 0, 255), "violet": Color(148, 0, 211),
    "magenta": Color(255, 0, 255), "pink": Color(255, 60, 120), "cyan": Color(0, 255, 255),
    "teal": Color(0, 180, 180), "lime": Color(120, 255, 0), "warm": Color(255, 140, 40),
    "cool": Color(120, 150, 190),
}

_DUR_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
}


def parse_color(token: str) -> Color | None:
    t = token.strip().lower()
    if not t:
        return None
    if t in NAMED_COLORS:
        return NAMED_COLORS[t]
    bird = birds.get(t)
    if bird is not None:
        return bird.color
    hexpart = t[1:] if t.startswith("#") else t
    if re.fullmatch(r"[0-9a-f]{6}", hexpart):
        return Color.from_hex(hexpart)
    if re.fullmatch(r"[0-9a-f]{3}", hexpart):
        return Color.from_hex("".join(c * 2 for c in hexpart))
    return None


def parse_duration(text: str, bare_unit: float = 60.0) -> float | None:
    """Parse '10 mins', '10m', '30s', '1h', or a bare number.

    A bare number (no unit) is multiplied by ``bare_unit`` seconds — default 60
    (minutes, matching the /rgb color default), but callers like `/rgb test` pass
    bare_unit=1.0 so a bare number means seconds.
    """
    t = text.strip().lower()
    if not t:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]*)", t)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "":
        return value * bare_unit
    mult = _DUR_UNITS.get(unit)
    return value * mult if mult else None


class RGBController:
    def __init__(
        self,
        stop_event: threading.Event,
        *,
        surface: Surface | None = None,
        layout: Layout | None = None,
        brightness: float = 1.0,
        fps: float = 30.0,
        ir_state=None,
        surface_config: SurfaceConfig | None = None,
        night_led: int = -1,
    ) -> None:
        self._stop = stop_event
        self._own_stop = threading.Event()  # so stop() can blank even if shared event not yet set
        self._lock = threading.Lock()
        self._brightness = max(0.0, min(1.0, brightness))
        self._fps = max(5.0, fps)
        self._ir_state = ir_state

        self._surface = surface or OpenRGBSurface(surface_config or SurfaceConfig())
        self._layout = layout or load_layout(self._surface, night_led=night_led)

        self._state = SceneState()
        self._queue: list[str] = []  # detection stack: day-bird labels, newest first
        self._night_mode = False
        self._discovery_progress = None  # optional shared DiscoveryProgress to poll
        self._disc_was_active = False
        self._manual_color: Color | None = None
        self._manual_test = False
        self._manual_until = 0.0
        self._boot_start = 0.0
        self._thread: threading.Thread | None = None
        self._last_reconnect = 0.0
        self._reconnect_delay = RECONNECT_EVERY

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        self._boot_start = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="rgb-controller", daemon=True)
        self._thread.start()
        log.info("RGB controller started: %d LEDs, surface %s",
                 self._surface.size, "online" if self._surface.connected else "offline")

    def stop(self) -> None:
        self._own_stop.set()
        th = self._thread
        if th is not None:
            th.join(timeout=2.0)
        try:
            self._surface.clear()  # turn the lights off cleanly
        except Exception:  # noqa: BLE001
            pass

    def _stopping(self) -> bool:
        return self._stop.is_set() or self._own_stop.is_set()

    # -- state inputs (called from other threads) -----------------------------
    def on_discovery(self, snapshot: dict) -> None:
        """Fed the DiscoveryProgress.snapshot() dict on every sweep event."""
        try:
            counts = snapshot.get("counts", {})
            total = len(snapshot.get("order", [])) or 1
            found = int(counts.get("found", 0))
            failed = int(counts.get("failed", 0))
            active = bool(snapshot.get("active", False))
        except Exception:  # noqa: BLE001
            return
        now = time.monotonic()
        with self._lock:
            st = self._state
            if found > st.discovery_found:
                st.last_found_at = now
            st.discovery_found = found
            st.discovery_total = total
            st.discovery_fraction = (found + failed) / total
            st.discovery_active = active
            if not active:
                st.discovery_ended_at = now

    def bind_discovery(self, progress) -> None:
        """Poll a shared DiscoveryProgress each tick instead of needing a callback
        threaded through every discover_and_apply() call site."""
        self._discovery_progress = progress

    def _poll_discovery(self) -> None:
        p = self._discovery_progress
        if p is None:
            return
        try:
            active = p.is_active()
            if active or self._disc_was_active:
                # feed while active, and exactly once on the active->inactive edge
                self.on_discovery(p.snapshot())
            self._disc_was_active = active
        except Exception:  # noqa: BLE001
            pass

    def on_detection(self, camera: str, detections, is_ir: bool) -> None:
        """Fed each inference's detections (Detection has .label/.confidence)."""
        now = time.monotonic()
        with self._lock:
            for det in detections:
                label = getattr(det, "label", "").lower()
                conf = float(getattr(det, "confidence", 1.0))
                bird = birds.get(label)
                if is_ir:
                    if bird is not None and bird.night:
                        self._state.night = Sighting(label, bird.color, conf, now, now)
                else:
                    if bird is not None and not bird.night:
                        prev = self._state.day.get(label)
                        first = prev.first_seen if prev else now
                        self._state.day[label] = Sighting(label, bird.color, conf, now, first)
                        # Push a newly-appeared bird onto the front of the stack;
                        # a bird that's already showing keeps its position (just
                        # its timestamp refreshes above), so co-present birds don't
                        # jitter every frame.
                        if label not in self._queue:
                            self._queue.insert(0, label)

    def on_ir(self, camera: str, is_ir: bool) -> None:
        """IRState listener: recompute night mode from the whole flock."""
        with self._lock:
            if self._ir_state is not None:
                try:
                    self._night_mode = self._ir_state.all_ir()
                    return
                except Exception:  # noqa: BLE001
                    pass
            self._night_mode = is_ir

    def set_error(self, message: str | None) -> None:
        with self._lock:
            self._state.error = message
            if message:
                self._state.error_at = time.monotonic()

    # -- /rgb command ---------------------------------------------------------
    def command(self, argument: str) -> str:
        arg = (argument or "").strip()
        if not arg or arg.lower() in ("status", "info"):
            return self.status_text()
        tokens = arg.split()
        head = tokens[0].lower()

        if head == "auto":
            with self._lock:
                self._manual_color = None
                self._manual_test = False
                self._manual_until = 0.0
            return "🌈 RGB: auto mode — lights follow the aviary (discovery / birds / night)."

        if head in ("calibrate", "calibration"):
            return ("To map each physical LED to a bird, run on the host:\n"
                    "    uv run rgb-calibrate")

        if head == "test":
            parsed = parse_duration(" ".join(tokens[1:]), bare_unit=1.0)  # test secs
            secs = 12.0 if parsed is None else parsed
            with self._lock:
                self._manual_test = True
                self._manual_color = None
                self._manual_until = time.monotonic() + secs
            return f"🎆 RGB: test pattern for {_fmt_secs(secs)} (use /rgb auto to stop)."

        color = parse_color(head)
        if color is None:
            return (f"Unknown color '{head}'. Try a name (red, green, blue, white, gold…), "
                    "a bird (draft, pizza…), or #RRGGBB. '/rgb auto' resumes context mode.")
        secs = DEFAULT_MANUAL_SECONDS
        if len(tokens) > 1:
            parsed = parse_duration(" ".join(tokens[1:]))
            if parsed is None:
                return f"Couldn't read the duration '{' '.join(tokens[1:])}'. Try '10m', '30s', '1h'."
            secs = parsed
        with self._lock:
            self._manual_color = color
            self._manual_test = False
            self._manual_until = time.monotonic() + secs
        name = head if head in NAMED_COLORS or birds.get(head) else color.to_hex()
        return f"🎨 RGB: {name} for {_fmt_secs(secs)}, then back to auto."

    def status_text(self) -> str:
        with self._lock:
            now = time.monotonic()
            manual = self._manual_color is not None or self._manual_test
            remaining = max(0.0, self._manual_until - now) if manual else 0.0
            night = self._night_mode
            disc = self._state.discovery_active
            # stack order: newest first (front of the strip)
            day = [lbl for lbl in self._queue
                   if lbl in self._state.day
                   and scenes._decay(now, self._state.day[lbl].last_seen, scenes.DETECTION_TTL) > 0]
        surf = self._surface
        lines = ["💡 RGB status"]
        if manual:
            what = "test pattern" if self._manual_test else (self._manual_color.to_hex() if self._manual_color else "?")
            lines.append(f"  mode: MANUAL ({what}), {_fmt_secs(remaining)} left → auto")
        elif disc:
            lines.append("  mode: DISCOVERY (loading wave)")
        elif night:
            lines.append("  mode: NIGHT (dark; night LED on sightings)")
        elif day:
            lines.append(f"  mode: DETECTION — {', '.join(day)}")
        else:
            lines.append("  mode: IDLE (breathing)")
        lines.append(f"  surface: {surf.size} LEDs, {'online' if surf.connected else 'OFFLINE'}")
        lines.append(f"  brightness: {int(self._brightness * 100)}%")
        return "\n".join(lines)

    def _maybe_reconnect(self, t: float) -> None:
        if (t - self._last_reconnect) <= self._reconnect_delay:
            return
        self._last_reconnect = t
        if self._surface.reconnect():
            self._reconnect_delay = RECONNECT_EVERY
        else:
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX)

    # -- render loop ----------------------------------------------------------
    def _run(self) -> None:
        period = 1.0 / self._fps
        while not self._stopping():
            t = time.monotonic()
            try:
                self._poll_discovery()
                if not self._surface.connected:
                    # No hardware to drive: skip the render entirely and probe
                    # for the server on the backoff schedule.
                    self._maybe_reconnect(t)
                else:
                    frame = self._render(t)
                    if self._brightness < 1.0:
                        frame = [c.scale(self._brightness) for c in frame]
                    if not self._surface.write(frame):
                        self._maybe_reconnect(t)
            except Exception:  # noqa: BLE001
                log.exception("RGB render tick failed")
            if self._own_stop.wait(period) or self._stop.wait(0):
                break
        # blanked by stop()

    def _render(self, t: float) -> list[Color]:
        with self._lock:
            st = self._state
            manual_color = self._manual_color
            manual_test = self._manual_test
            manual_until = self._manual_until
            night = self._night_mode
            # expire manual override
            if (manual_color is not None or manual_test) and t >= manual_until:
                self._manual_color = manual_color = None
                self._manual_test = manual_test = False

            # Prune the detection stack: drop birds that have decayed out, keeping
            # the rest in order so the remaining ones "shift up" to fill the gap.
            self._queue = [
                lbl for lbl in self._queue
                if lbl in st.day and scenes._decay(t, st.day[lbl].last_seen, scenes.DETECTION_TTL) > 0
            ]
            queue_copy = list(self._queue)

            # Snapshot scene state under the lock so the scene functions can read
            # it without racing on_detection()/on_discovery() mutating it from the
            # camera + discovery threads. Sightings are replaced (never mutated in
            # place), so a shallow copy of the day dict is safe.
            state_copy = SceneState(
                day=dict(st.day),
                queue=queue_copy,
                night=st.night,
                discovery_active=st.discovery_active,
                discovery_fraction=st.discovery_fraction,
                discovery_total=st.discovery_total,
                discovery_found=st.discovery_found,
                last_found_at=st.last_found_at,
                discovery_ended_at=st.discovery_ended_at,
                error=st.error,
                error_at=st.error_at,
            )
            layout = self._layout

        day_active = bool(queue_copy)
        disc_active = state_copy.discovery_active or (
            t - state_copy.discovery_ended_at
        ) < scenes.DISCOVERY_LINGER
        error = state_copy.error

        # MANUAL override (explicit user intent) wins.
        if manual_test:
            return scenes.party(t, layout)
        if manual_color is not None:
            return scenes.solid(manual_color, layout)
        # BOOT self-test
        if (t - self._boot_start) < BOOT_SECONDS:
            return scenes.boot(t, layout, BOOT_SECONDS)
        if error:
            return scenes.error(state_copy, t, layout)
        if disc_active:
            return scenes.discovery(state_copy, t, layout)
        if night:
            return scenes.night(state_copy, t, layout)
        if day_active:
            return scenes.detection(state_copy, t, layout)
        return scenes.idle(t, layout)


def _fmt_secs(secs: float) -> str:
    secs = int(round(secs))
    if secs >= 3600:
        h, rem = divmod(secs, 3600)
        m = rem // 60
        return f"{h}h{m:02d}m" if m else f"{h}h"
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{m}m{s:02d}s" if s else f"{m}m"
    return f"{secs}s"
