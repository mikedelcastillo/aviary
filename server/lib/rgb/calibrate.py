"""Interactive LED calibration for the detection STACK.

Detection isn't one-LED-per-bird — the whole strip is a recency stack (newest
bird at the front, older ones shifted back). So calibration just defines:
  * the stack ORDER — which physical LED is the front, and the flow after it,
  * the single NIGHT LED used in IR mode,
  * (optionally) the pixel count of an addressable header/fan.

You assign each LED a position by lighting it and entering its rank (1 = front).
Anything you skip is appended after the ranked ones; 'x' drops an LED entirely.

Run on the host (the LEDs have to be visible):
    uv run rgb-calibrate
"""
from __future__ import annotations

import sys

from .layout import DEFAULT_LAYOUT_PATH, Layout
from .palette import BLACK, Color, WHITE
from .surface import OpenRGBSurface, SurfaceConfig

HILITE = WHITE


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "quit"


def _light_only(surface: OpenRGBSurface, index: int, color: Color = HILITE) -> None:
    frame = [BLACK] * surface.size
    if 0 <= index < surface.size:
        frame[index] = color
    surface.write(frame)


def main() -> None:
    print("=== aviary RGB calibration (detection stack) ===")
    host, port = "127.0.0.1", 6742
    surface = OpenRGBSurface(SurfaceConfig(host=host, port=port))
    if not surface.connected:
        print(f"Could not reach the OpenRGB server at {host}:{port}.")
        print("Start it first (the installer sets up a systemd user service):")
        print("    systemctl --user start aviary-openrgb")
        sys.exit(1)

    print(surface.describe())
    print()

    # Optionally size an addressable header (e.g. an ARGB fan ring) so its pixels
    # join the stack.
    if any(li.zone_addressable for li in surface.leds):
        print("Tip: set an addressable strip/fan's pixel count to add its LEDs to the stack.")
        ans = _prompt("Addressable strip pixel count to apply now (blank to skip): ")
        if ans.isdigit() and int(ans) >= 0:
            surface = OpenRGBSurface(SurfaceConfig(host=host, port=port, addressable_len=int(ans)))
            print(f"Re-enumerated with addressable_len={ans} ({surface.size} LEDs)."
                  f" Persist it with RGB_STRIP_LEN={ans} in .env.\n")

    print("For each LED I'll light it WHITE. Enter its STACK POSITION:")
    print("  a number (1 = front/top of the stack, where the newest bird shows),")
    print("  [Enter] = include after the ranked LEDs, 'night' = the IR night LED,")
    print("  'x' = exclude this LED, 'quit' = finish now.\n")

    ranked: list[tuple[int, int]] = []   # (rank, led_index)
    unranked: list[int] = []
    excluded: set[int] = set()
    night_led: int | None = None

    for li in surface.leds:
        _light_only(surface, li.index)
        while True:
            ans = _prompt(f"LED {li.index} [{li.zone}] -> ").lower()
            if ans in ("quit", "q"):
                break
            if ans == "":
                unranked.append(li.index)
                break
            if ans in ("x", "exclude", "off"):
                excluded.add(li.index)
                break
            if ans in ("night", "n"):
                night_led = li.index
                print(f"  ★ night LED = {li.index}")
                break
            if ans.isdigit():
                ranked.append((int(ans), li.index))
                print(f"  ✓ LED {li.index} -> stack position {ans}")
                break
            print("  Enter a number (1=front), 'night', 'x', blank to append, or 'quit'.")
        if ans in ("quit", "q"):
            break

    bar = [idx for _, idx in sorted(ranked, key=lambda r: r[0])]
    bar += [i for i in unranked if i not in excluded]
    if not bar:
        bar = [li.index for li in surface.leds if li.index not in excluded]
    if night_led is None:
        night_led = bar[-1] if bar else 0

    layout = Layout(size=surface.size, bar=bar, night_led=night_led).clamp()
    layout.save(DEFAULT_LAYOUT_PATH)

    surface.clear()
    print()
    print(f"Saved stack layout to {DEFAULT_LAYOUT_PATH}")
    print(f"  stack order (front → back): {layout.bar}")
    print(f"  night LED: {layout.night_led}")
    print("\nDone. Restart the aviary server (or it picks this up on next start).")


if __name__ == "__main__":
    main()
