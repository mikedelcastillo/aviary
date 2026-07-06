"""Color model and math for the LED status display.

Deliberately dependency-free (no numpy) so the RGB subsystem can run in the same
slim process as the stdlib-only discovery sweep, and degrade to a no-op when the
ML stack or the OpenRGB server is absent.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

# Perceptual gamma. LEDs are linear in PWM duty but the eye is not; applying a
# gamma on the way out makes dim values (breathing tails, loading-bar trails)
# look smooth instead of clipping to off.
GAMMA = 2.2


@dataclass(frozen=True)
class Color:
    """An 8-bit-per-channel RGB color. Immutable so scenes can share constants."""

    r: int
    g: int
    b: int

    # -- constructors ---------------------------------------------------------
    @staticmethod
    def from_hex(s: str) -> "Color":
        s = s.lstrip("#")
        return Color(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

    @staticmethod
    def from_hsv(h: float, s: float, v: float) -> "Color":
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
        return Color(round(r * 255), round(g * 255), round(b * 255))

    # -- conversions ----------------------------------------------------------
    def to_hex(self) -> str:
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def hsv(self) -> tuple[float, float, float]:
        return colorsys.rgb_to_hsv(self.r / 255, self.g / 255, self.b / 255)

    # -- operations -----------------------------------------------------------
    def scale(self, factor: float) -> "Color":
        """Multiply brightness by ``factor`` in gamma-correct (linear) space."""
        factor = max(0.0, factor)

        def ch(v: int) -> int:
            lin = (v / 255.0) ** GAMMA
            out = (lin * factor) ** (1.0 / GAMMA)
            return max(0, min(255, round(out * 255)))

        return Color(ch(self.r), ch(self.g), ch(self.b))

    def blend(self, other: "Color", t: float) -> "Color":
        """Linear crossfade toward ``other`` by ``t`` in [0,1] (gamma-correct)."""
        t = max(0.0, min(1.0, t))

        def ch(a: int, b: int) -> int:
            la = (a / 255.0) ** GAMMA
            lb = (b / 255.0) ** GAMMA
            out = (la * (1 - t) + lb * t) ** (1.0 / GAMMA)
            return max(0, min(255, round(out * 255)))

        return Color(ch(self.r, other.r), ch(self.g, other.g), ch(self.b, other.b))

    def with_value(self, v: float) -> "Color":
        """Return this hue/sat at brightness ``v`` (HSV value, 0..1)."""
        h, s, _ = self.hsv()
        return Color.from_hsv(h, s, v)

    def is_dark(self, threshold: int = 8) -> bool:
        return max(self.r, self.g, self.b) <= threshold


# Common constants
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
RED = Color(255, 0, 0)


def pulse(t: float, low: float = 0.15, high: float = 1.0, period: float = 2.0) -> float:
    """A smooth cosine breathing envelope in [low, high] with the given period (s)."""
    import math

    phase = (t % period) / period
    # cosine eased 0->1->0
    e = (1 - math.cos(2 * math.pi * phase)) / 2
    return low + (high - low) * e


def heartbeat(t: float, period: float = 1.2) -> float:
    """Two quick bumps then rest — a literal heartbeat envelope in [0,1]."""
    import math

    phase = (t % period) / period
    if phase < 0.12:
        return math.sin(phase / 0.12 * math.pi)
    if 0.18 < phase < 0.30:
        return 0.7 * math.sin((phase - 0.18) / 0.12 * math.pi)
    return 0.06
