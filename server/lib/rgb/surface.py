"""The LED *surface*: a flat, ordered array of individually-addressable LEDs that
the scene engine paints, decoupled from how the pixels physically reach the board.

Two implementations:
  * ``OpenRGBSurface`` — talks to a running OpenRGB SDK server (127.0.0.1:6742),
    sets every device to Direct mode, and (optionally) resizes addressable
    headers (e.g. the D_LED ARGB fan) so their pixels become controllable.
  * ``VirtualSurface`` — an in-memory surface with no hardware, used by the
    visualizer/simulator and as the graceful fallback when no server is reachable.

The engine never crashes if OpenRGB is missing: construction falls back to a
disconnected surface whose ``write`` is a no-op, and ``reconnect()`` is retried.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from .palette import BLACK, Color

log = logging.getLogger("aviary.rgb.surface")


@dataclass(frozen=True)
class LedInfo:
    """Metadata for one physical LED in the flattened surface."""

    index: int          # global index in the surface
    device: int         # OpenRGB device index
    device_name: str
    zone: str
    zone_addressable: bool
    name: str


@dataclass
class SurfaceConfig:
    host: str = "127.0.0.1"
    port: int = 6742
    client_name: str = "aviary"
    # Set every addressable zone (e.g. the D_LED ARGB header / fan ring) to this
    # many pixels. -1 = leave at OpenRGB's detected size; 0 = force OFF (disable
    # the header); N = N pixels.
    addressable_len: int = -1
    # Explicit per-zone pixel counts, keyed "Device Name/Zone Name" — wins over
    # addressable_len. Populated by calibration once the real strip length is known.
    zone_sizes: dict[str, int] = field(default_factory=dict)


class Surface:
    """Base class: a fixed-length ordered list of LEDs plus a last-frame buffer."""

    def __init__(self) -> None:
        self.leds: list[LedInfo] = []
        self.last_frame: list[Color] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self.leds)

    @property
    def connected(self) -> bool:
        return False

    def write(self, colors: list[Color]) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def clear(self) -> None:
        self.write([BLACK] * self.size)

    def reconnect(self) -> bool:
        return self.connected

    def describe(self) -> str:
        lines = [f"surface: {self.size} LEDs ({'connected' if self.connected else 'offline'})"]
        last_dev = None
        for li in self.leds:
            if li.device_name != last_dev:
                lines.append(f"  device {li.device}: {li.device_name}")
                last_dev = li.device_name
            tag = " [addressable]" if li.zone_addressable else ""
            lines.append(f"    [{li.index:>3}] {li.zone}{tag}: {li.name}")
        return "\n".join(lines)


class VirtualSurface(Surface):
    """In-memory surface for the simulator / no-hardware fallback."""

    def __init__(self, leds: list[LedInfo] | None = None, size: int = 0) -> None:
        super().__init__()
        if leds is not None:
            self.leds = leds
        else:
            self.leds = [
                LedInfo(i, 0, "Virtual", "virtual", False, f"led{i}") for i in range(size)
            ]
        self.last_frame = [BLACK] * self.size

    @property
    def connected(self) -> bool:
        return True  # the virtual surface is always "available"

    def write(self, colors: list[Color]) -> bool:
        with self._lock:
            n = self.size
            self.last_frame = (list(colors) + [BLACK] * n)[:n]
        return True


class OpenRGBSurface(Surface):
    """Drives real hardware through the OpenRGB SDK server."""

    def __init__(self, config: SurfaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or SurfaceConfig()
        self._client = None
        self._devices: list = []          # openrgb Device objects
        self._spans: list[tuple[int, int, int]] = []  # (device_idx_in_list, start, end)
        self._RGBColor = None
        self._offline_logged = False
        self.reconnect()

    @property
    def connected(self) -> bool:
        return self._client is not None

    # -- connection -----------------------------------------------------------
    def reconnect(self) -> bool:
        if self.connected:
            return True
        try:
            from openrgb import OpenRGBClient
            from openrgb.utils import RGBColor
        except Exception as e:  # noqa: BLE001 - openrgb-python not installed
            log.info("openrgb-python unavailable (%s); RGB output disabled.", e)
            return False
        try:
            self._RGBColor = RGBColor
            self._client = OpenRGBClient(
                address=self.config.host, port=self.config.port, name=self.config.client_name
            )
            self._build()
            log.info("OpenRGB connected: %d LEDs across %d device(s).", self.size, len(self._devices))
            self._offline_logged = False
            return True
        except Exception as e:  # noqa: BLE001 - server down / refused
            # First miss at INFO; the retries drop to DEBUG so a box without an
            # OpenRGB server doesn't fill the log with an identical line forever.
            level = logging.DEBUG if self._offline_logged else logging.INFO
            log.log(level, "OpenRGB server not reachable at %s:%s (%s); will retry.",
                    self.config.host, self.config.port, e)
            self._offline_logged = True
            self._client = None
            return False

    def _build(self) -> None:
        """Enumerate devices, force Direct mode, resize addressable zones, flatten."""
        from openrgb.utils import ModeData  # noqa: F401  (ensure utils import works)

        leds: list[LedInfo] = []
        spans: list[tuple[int, int, int]] = []
        self._devices = list(self._client.devices)
        gi = 0
        for di, dev in enumerate(self._devices):
            self._force_direct(dev)
            self._resize_addressable(dev)
            addressable_zone_leds: set[int] = set()
            cursor = 0
            for z in dev.zones:
                z_is_addr = _zone_is_addressable(z)
                for _led in z.leds:
                    addressable_zone_leds.add(cursor) if z_is_addr else None
                    cursor += 1
            start = gi
            for li, led in enumerate(dev.leds):
                leds.append(LedInfo(
                    index=gi, device=di, device_name=dev.name,
                    zone=_led_zone_name(dev, li),
                    zone_addressable=(li in addressable_zone_leds),
                    name=getattr(led, "name", f"led{li}"),
                ))
                gi += 1
            spans.append((di, start, gi))
        self.leds = leds
        self._spans = spans
        self.last_frame = [BLACK] * self.size

    def _force_direct(self, dev) -> None:
        try:
            if any(m.name.lower() == "direct" for m in dev.modes):
                dev.set_mode("Direct")
        except Exception as e:  # noqa: BLE001
            log.debug("set Direct failed on %s: %s", getattr(dev, "name", "?"), e)

    def _resize_addressable(self, dev) -> None:
        for z in dev.zones:
            if not _zone_is_addressable(z):
                continue
            key = f"{dev.name}/{z.name}"
            # -1 (the sentinel) means "leave at the detected size"; 0 forces the
            # header OFF; any N>=0 sets exactly N pixels.
            target = self.config.zone_sizes.get(key, self.config.addressable_len)
            try:
                if target is not None and target >= 0 and len(z.leds) != target:
                    z.resize(target)
                    log.info("resized addressable zone %s -> %d LEDs", key, target)
            except Exception as e:  # noqa: BLE001
                log.debug("resize %s failed: %s", key, e)

    # -- output ---------------------------------------------------------------
    def write(self, colors: list[Color]) -> bool:
        if not self.connected:
            return False
        n = self.size
        frame = (list(colors) + [BLACK] * n)[:n]
        RGB = self._RGBColor
        # Serialize ALL device writes under the lock: the render thread and a
        # stop()/clear() from another thread share one (non-thread-safe) OpenRGB
        # socket, so they must not interleave — and this guarantees clear()'s
        # blank lands strictly after any in-flight render write on shutdown.
        with self._lock:
            try:
                for di, start, end in self._spans:
                    dev = self._devices[di]
                    dev_colors = [RGB(c.r, c.g, c.b) for c in frame[start:end]]
                    if dev_colors:
                        dev.set_colors(dev_colors, fast=True)
                self.last_frame = frame
                return True
            except Exception as e:  # noqa: BLE001 - server dropped mid-write
                log.warning("OpenRGB write failed (%s); marking disconnected.", e)
                self._client = None
                return False


def _zone_is_addressable(z) -> bool:
    """True for LINEAR/MATRIX zones (resizable strips/rings) vs SINGLE zones."""
    try:
        from openrgb.utils import ZoneType
        return z.type in (ZoneType.LINEAR, ZoneType.MATRIX)
    except Exception:  # noqa: BLE001
        return str(getattr(z, "type", "")) in ("1", "2")


def _led_zone_name(dev, led_index: int) -> str:
    """Map a device-flat LED index back to its owning zone's name."""
    cursor = 0
    for z in dev.zones:
        count = len(z.leds)
        if cursor <= led_index < cursor + count:
            return z.name
        cursor += count
    return "?"
