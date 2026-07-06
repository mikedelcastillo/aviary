"""Logical → physical LED mapping.

A ``Layout`` answers three questions for the scene engine:
  * which physical LED is a given bird's "home" slot (``slots``),
  * in what order do LEDs form the loading-bar / wave (``bar``),
  * which single LED is the night LED (``night_led``).

It auto-builds a sensible default from whatever the surface enumerates, and is
overridden by a calibration file written by ``rgb-calibrate`` once the user has
told us which physical light is which. Everything degrades: with only 5 LEDs the
day birds share the bar dynamically; with a 9+ LED ARGB strip each bird gets a
dedicated home.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .surface import Surface

log = logging.getLogger("aviary.rgb.layout")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = _REPO_ROOT / "data" / "server" / "rgb_layout.json"
LAYOUT_VERSION = 1


@dataclass
class Layout:
    """How the detection stack maps onto physical LEDs.

    ``bar`` is the ordered list of physical LED indices the stack flows along
    (index 0 = the front/top of the stack, where the newest bird shows).
    ``night_led`` is the single LED used in IR/night mode.
    """

    size: int
    bar: list[int] = field(default_factory=list)
    night_led: int = 0

    def clamp(self) -> "Layout":
        self.bar = [i for i in self.bar if 0 <= i < self.size]
        if not self.bar:
            self.bar = list(range(self.size))
        if not (0 <= self.night_led < self.size):
            self.night_led = max(0, self.size - 1)
        return self

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": LAYOUT_VERSION,
            "size": self.size,
            "bar": self.bar,
            "night_led": self.night_led,
        }

    def save(self, path: Path | str = DEFAULT_LAYOUT_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("Saved RGB layout to %s", p)


def default_layout(surface: Surface, night_led: int = -1) -> Layout:
    """Auto-build: the stack flows in surface order; the last LED is the night LED."""
    size = surface.size
    bar = list(range(size))
    nl = night_led if night_led >= 0 else max(0, size - 1)
    return Layout(size=size, bar=bar, night_led=nl).clamp()


def load_layout(
    surface: Surface, path: Path | str = DEFAULT_LAYOUT_PATH, night_led: int = -1
) -> Layout:
    """Load a saved stack layout (bar order + night LED) if present, else default."""
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            lay = Layout(
                size=surface.size,
                bar=list(data.get("bar", [])),
                night_led=int(data.get("night_led", max(0, surface.size - 1))),
            ).clamp()
            saved_size = int(data.get("size", surface.size))
            if saved_size != surface.size:
                log.warning(
                    "RGB layout was saved for %d LEDs but surface now has %d; "
                    "using it where valid (re-run rgb-calibrate to refresh).",
                    saved_size, surface.size,
                )
            log.info("Loaded RGB layout from %s (bar of %d)", p, len(lay.bar))
            return lay
        except Exception:  # noqa: BLE001
            log.exception("Failed to read RGB layout %s; using default.", p)
    return default_layout(surface, night_led=night_led)
