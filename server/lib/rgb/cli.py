"""CLI entrypoints: `uv run rgb-demo` and `uv run rgb-status`.

`rgb-demo` exercises the *real* engine with synthetic aviary events (boot →
discovery sweep → birds → full-flock party → night) so you can watch the actual
scene pipeline on the LEDs (or in the terminal simulator with --sim).
"""
from __future__ import annotations

import sys
import threading
import time

from . import birds, visualizer
from .engine import RGBController
from .scenes import Sighting
from .surface import LedInfo, OpenRGBSurface, SurfaceConfig, VirtualSurface


def _make_surface(sim: bool):
    if not sim:
        surf = OpenRGBSurface(SurfaceConfig())
        if surf.connected:
            return surf, False
        print("OpenRGB server not reachable — falling back to the terminal simulator.\n")
    # virtual 9-LED board mirrors the supplied roster legend
    leds = [LedInfo(i, 0, "Virtual", "roster", i >= 4, f"led{i}") for i in range(9)]
    return VirtualSurface(leds=leds), True


def _snap(found: int, failed: int, total: int, active: bool) -> dict:
    states = {}
    order = [f"10.0.0.{i+1}" for i in range(total)]
    for i in range(found):
        states[order[i]] = "found"
    for i in range(found, found + failed):
        states[order[i]] = "failed"
    return {
        "active": active, "network": "10.0.0.", "order": order, "states": states,
        "counts": {"pending": total - found - failed, "testing": 0, "found": found, "failed": failed},
    }


def demo() -> None:
    sim = "--sim" in sys.argv
    surface, virtual = _make_surface(sim)
    stop = threading.Event()
    ctrl = RGBController(stop, surface=surface, fps=30.0)

    def show(label: str, secs: float) -> None:
        print(f"▶ {label}")
        end = time.monotonic() + secs
        while time.monotonic() < end:
            if virtual:
                sys.stdout.write("\033[H\033[2J" + visualizer.render(surface) + "\n")
                sys.stdout.flush()
            time.sleep(1 / 15 if virtual else 0.1)

    ctrl.start()
    try:
        show("BOOT self-test", 4.5)

        # discovery sweep
        total = 24
        for i in range(total + 1):
            found = 1 if i >= 8 else 0
            found += 1 if i >= 16 else 0
            ctrl.on_discovery(_snap(found, i - found, total, active=True))
            show(f"DISCOVERY {i}/{total}", 0.18)
        ctrl.on_discovery(_snap(2, total - 2, total, active=False))
        show("DISCOVERY complete", 2.0)

        # individual birds, one at a time (watch the personalities)
        for b in birds.INDIVIDUALS:
            ctrl.on_detection("cam1", [_Det(b.name, 0.9)], is_ir=False)
            show(f"DETECTION: {b.name} ({b.personality})", 3.0)

        # full flock => party
        for b in birds.INDIVIDUALS:
            ctrl.on_detection("cam1", [_Det(b.name, 0.95)], is_ir=False)
        show("FULL-FLOCK PARTY 🎉", 5.0)
        time.sleep(7)  # let day sightings decay

        # night
        ctrl.on_ir("cam1", True)
        show("NIGHT (dark)", 2.5)
        ctrl.on_detection("cam1", [_Det("lovebird", 0.8)], is_ir=True)
        show("NIGHT: lovebird seen", 4.0)
        ctrl.on_ir("cam1", False)
        show("back to IDLE", 3.0)
    finally:
        ctrl.stop()
    print("demo complete.")


def status() -> None:
    surface = OpenRGBSurface(SurfaceConfig())
    if not surface.connected:
        print("OpenRGB server not reachable at 127.0.0.1:6742.")
        print("Start it:  systemctl --user start aviary-openrgb")
        sys.exit(1)
    print(surface.describe())
    warnings = _roster_warnings()
    if warnings:
        print("\nroster warnings:")
        for w in warnings:
            print("  ! " + w)


def _roster_warnings() -> list[str]:
    try:
        from lib.roster import load_species_members
        return birds.validate_against(load_species_members())
    except Exception:  # noqa: BLE001
        return []


class _Det:
    """Minimal stand-in for detector.Detection (label/confidence)."""

    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence
        self.bbox_xyxy = (0, 0, 0, 0)


if __name__ == "__main__":
    demo()
