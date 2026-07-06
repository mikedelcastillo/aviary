"""Scene renderers — pure functions that paint a frame (list[Color]) from state.

Each scene takes the shared :class:`SceneState`, the wall-clock ``t`` (monotonic
seconds), and the :class:`Layout`, and returns a full-length list of Colors. The
engine picks which scene is active by priority and writes the result.

Design intent (the "fun"): the PC becomes a living readout of the aviary.
  * boot      — a KITT chase so you know the server just came up.
  * discovery — a white loading-wave that fills as the network sweep completes,
                with a green pop each time a camera is confirmed.
  * detection — a recency STACK across the whole strip: the newest bird's color
                at the front, older sightings shifted one LED back (dimmer), the
                oldest falling off the end; each keeps its personality pulse. All
                six birds at once trips a rainbow "full-flock" celebration.
  * night     — everything dark except one LED that glows the color of whichever
                night species the IR cameras just saw.
  * idle      — a slow calm breath.
  * error     — a red pulse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import birds
from .layout import Layout
from .palette import BLACK, RED, WHITE, Color, pulse

# Time-to-live / fade windows (seconds)
DETECTION_TTL = 6.0      # a day bird's LED fades out this long after last seen
NIGHT_TTL = 8.0          # the night LED fades this long after a night sighting
FOUND_FLASH = 1.2        # green pop duration after a camera is confirmed
DISCOVERY_LINGER = 2.5   # keep the full bar lit this long after sweep ends
# (the full-flock party triggers when all day birds are present within DETECTION_TTL)


@dataclass
class Sighting:
    label: str
    color: Color
    confidence: float
    last_seen: float
    first_seen: float


@dataclass
class SceneState:
    # detection
    day: dict[str, Sighting] = field(default_factory=dict)
    # The detection stack: ordered list of day-bird labels, newest at index 0.
    # The engine pushes a bird to the front on first sighting and prunes it when
    # it decays, so the whole strip renders as a shifting queue of recent birds.
    queue: list[str] = field(default_factory=list)
    night: Sighting | None = None
    # discovery
    discovery_active: bool = False
    discovery_fraction: float = 0.0
    discovery_total: int = 0
    discovery_found: int = 0
    last_found_at: float = -1e9
    discovery_ended_at: float = -1e9
    # misc
    error: str | None = None
    error_at: float = -1e9


# --- envelopes ---------------------------------------------------------------
def personality_env(personality: str, t: float, seed: float) -> float:
    """A 0..1 brightness envelope giving each bird a recognizable rhythm."""
    if personality == "steady":
        return 1.0
    if personality == "breathe":
        return pulse(t, 0.35, 1.0, 2.6)
    if personality == "flutter":
        return 0.72 + 0.28 * (0.5 + 0.5 * math.sin(t * 9.0 + seed))
    if personality == "twinkle":
        s = 0.5 + 0.5 * math.sin(t * 3.0 + seed)
        return 0.45 + 0.55 * (s ** 3)
    if personality == "bounce":
        phase = (t * 1.6 + seed) % 1.0
        return 0.25 + 0.75 * (1.0 - phase) ** 2  # quick attack, slow fall
    return 1.0


def _decay(t: float, last_seen: float, ttl: float) -> float:
    return max(0.0, 1.0 - (t - last_seen) / ttl)


def _seed(name: str) -> float:
    return (sum(ord(c) for c in name) % 17) * 0.37


# --- scenes ------------------------------------------------------------------
def boot(t: float, layout: Layout, duration: float) -> list[Color]:
    """A bright dot sweeps back and forth across the bar with a fading trail."""
    n = layout.size
    frame = [BLACK] * n
    if n == 0:
        return frame
    bar = layout.bar
    m = len(bar)
    speed = max(1.0, m) / max(0.4, duration * 0.45)  # a couple of passes
    pos = (t * speed) % (2 * (m - 1)) if m > 1 else 0
    if pos >= m - 1:  # fold the triangle wave back at the last LED (symmetric)
        pos = 2 * (m - 1) - pos
    hue = (t * 0.5) % 1.0
    crest = Color.from_hsv(hue, 0.15, 1.0)
    for i, led in enumerate(bar):
        d = abs(i - pos)
        b = max(0.0, 1.0 - d / 2.2)
        if b > 0:
            frame[led] = crest.scale(b * b)
    return frame


def discovery(state: SceneState, t: float, layout: Layout) -> list[Color]:
    """White loading-wave that fills with the sweep; green pop on each new camera."""
    n = layout.size
    frame = [BLACK] * n
    bar = layout.bar
    m = len(bar)
    if m == 0:
        return frame

    frac = max(0.0, min(1.0, state.discovery_fraction))
    if not state.discovery_active:
        frac = 1.0  # lingering full bar after the sweep ends
    filled = frac * m
    found_flash = _decay(t, state.last_found_at, FOUND_FLASH)

    # base color: white, tinted green briefly when a camera was just found
    base = WHITE.blend(Color(0, 255, 60), 0.85 * found_flash)

    wave_speed = 2.2
    for i, led in enumerate(bar):
        if i < math.floor(filled):
            # behind the frontier: lit, with a travelling crest = the "wave"
            crest = 0.5 + 0.5 * math.sin((i / max(1, m)) * 6.28 - t * wave_speed)
            frame[led] = base.scale(0.30 + 0.45 * crest + 0.25 * found_flash)
        elif i < filled:
            # the frontier LED breathes brightly (the leading edge)
            frame[led] = base.scale(pulse(t, 0.5, 1.0, 0.7))
        else:
            frame[led] = base.scale(0.04)  # faint track ahead
    return frame


def detection(state: SceneState, t: float, layout: Layout) -> list[Color]:
    """Render the detection STACK across the whole strip.

    There is no fixed LED-per-bird mapping. ``state.queue`` is an ordered list of
    recent day birds (newest first, maintained by the engine): the front of the
    bar shows the most recent sighting, each older one sits one LED further back,
    and the oldest falls off the end. A new bird pushes onto the front and shifts
    everyone down. Older entries dim (a recency gradient) and each bird keeps its
    personality pulse. All six birds out at once => party.
    """
    n = layout.size
    frame = [BLACK] * n

    present = [
        lbl for lbl in state.queue
        if lbl in state.day and _decay(t, state.day[lbl].last_seen, DETECTION_TTL) > 0.0
    ]
    if not present:
        return frame

    # Easter egg: the whole flock is out at once.
    if len(set(present)) >= len(birds.DAY_ORDER):
        return party(t, layout)

    bar = layout.bar
    depth_n = max(1, len(bar))
    for i, label in enumerate(present[: len(bar)]):
        s = state.day[label]
        bird = birds.get(label)
        pers = bird.personality if bird else "breathe"
        env = personality_env(pers, t, _seed(label))
        decay = _decay(t, s.last_seen, DETECTION_TTL)
        depth = 1.0 - i / depth_n            # 1.0 at the front (newest) -> dimmer toward the back
        bright = (0.30 + 0.70 * s.confidence) * decay * env * (0.40 + 0.60 * depth)
        frame[bar[i]] = s.color.scale(bright)
    return frame


def party(t: float, layout: Layout) -> list[Color]:
    """Full-flock celebration: a rainbow that rolls along the bar."""
    n = layout.size
    frame = [BLACK] * n
    bar = layout.bar
    m = max(1, len(bar))
    for i, led in enumerate(bar):
        hue = (i / m + t * 0.6) % 1.0
        frame[led] = Color.from_hsv(hue, 1.0, pulse(t, 0.6, 1.0, 0.5))
    return frame


def night(state: SceneState, t: float, layout: Layout) -> list[Color]:
    """Everything dark except the single night LED, which glows the color of the
    night species last seen by the IR cameras (and fades when it leaves)."""
    frame = [BLACK] * layout.size
    s = state.night
    if s is None:
        return frame
    decay = _decay(t, s.last_seen, NIGHT_TTL)
    if decay <= 0.0:
        return frame
    idx = layout.night_led
    if 0 <= idx < layout.size:
        frame[idx] = s.color.scale(decay * pulse(t, 0.45, 1.0, 2.2))
    return frame


def idle(t: float, layout: Layout) -> list[Color]:
    """A slow, calm breath — the aviary at rest."""
    n = layout.size
    base = Color(120, 150, 190)  # soft cool white-blue
    b = pulse(t, 0.04, 0.22, 6.0)
    return [base.scale(b) for _ in range(n)]


def error(state: SceneState, t: float, layout: Layout) -> list[Color]:
    """Red alert pulse."""
    b = pulse(t, 0.15, 1.0, 0.9)
    return [RED.scale(b) for _ in range(layout.size)]


def solid(color: Color, layout: Layout) -> list[Color]:
    return [color] * layout.size
