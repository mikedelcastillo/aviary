"""Terminal mirror of the LED surface — a live "virtual board".

Doubles as the no-hardware simulator (point it at a VirtualSurface) and as a way
to see what the physical LEDs are doing from an SSH session. Uses 24-bit ANSI
color; falls back to plain text if the terminal doesn't support it.
"""
from __future__ import annotations

import os
import sys
import time

from .palette import Color
from .surface import Surface


def _supports_truecolor() -> bool:
    if not sys.stdout.isatty():
        return False
    return os.environ.get("COLORTERM", "") in ("truecolor", "24bit") or "256" in os.environ.get("TERM", "")


def _swatch(c: Color, truecolor: bool) -> str:
    if truecolor:
        return f"\033[48;2;{c.r};{c.g};{c.b}m  \033[0m"
    # crude fallback: brightness as a character
    level = max(c.r, c.g, c.b)
    ch = " .:-=+*#%@"[min(9, level // 26)]
    return ch * 2


def render(surface: Surface, *, truecolor: bool | None = None, width: int = 0) -> str:
    """Return a multi-line string showing each LED as a colored swatch + label."""
    tc = _supports_truecolor() if truecolor is None else truecolor
    frame = surface.last_frame or []
    lines = [f"board: {surface.size} LEDs  ({'online' if surface.connected else 'offline'})"]
    for li in surface.leds:
        c = frame[li.index] if li.index < len(frame) else Color(0, 0, 0)
        tag = "~" if li.zone_addressable else " "
        lines.append(
            f" {li.index:>3} {_swatch(c, tc)} {tag}{li.zone[:22]:<22} {c.to_hex()}"
        )
    return "\n".join(lines)


def render_strip(surface: Surface, *, truecolor: bool | None = None) -> str:
    """One-line swatch strip (compact), good for a live header."""
    tc = _supports_truecolor() if truecolor is None else truecolor
    frame = surface.last_frame or []
    cells = []
    for li in surface.leds:
        c = frame[li.index] if li.index < len(frame) else Color(0, 0, 0)
        cells.append(_swatch(c, tc))
    return "".join(cells)


def watch(surface: Surface, *, fps: float = 20.0, duration: float | None = None) -> None:
    """Live-redraw the board in the terminal until Ctrl-C or duration elapses."""
    period = 1.0 / max(1.0, fps)
    start = time.monotonic()
    try:
        sys.stdout.write("\033[2J")  # clear
        while True:
            sys.stdout.write("\033[H")  # home
            sys.stdout.write(render(surface) + "\n")
            sys.stdout.flush()
            if duration is not None and (time.monotonic() - start) >= duration:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
