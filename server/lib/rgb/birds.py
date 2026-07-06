"""The LED roster: every flock member's signature color + animation personality.

Colors are transcribed verbatim from the roster artwork the user supplied (the
9-dot legend): 6 living individuals above the divider, 3 IR/night species below.
This is the "roster" the lights speak in — see ``Bird`` below.

Keys are lowercase and must match ``training/roster.yaml`` label ``name`` fields
exactly; ``validate_against()`` cross-checks at startup so the map can't silently
drift when the flock changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .palette import Color

log = logging.getLogger("aviary.rgb.birds")


@dataclass(frozen=True)
class Bird:
    name: str
    color: Color
    # "personality": how this bird's LED animates when present. The point is that
    # you can tell *who* is in the room by the rhythm, not just the hue.
    personality: str
    species: str
    night: bool = False  # True for the 3 IR/species outline labels


# --- the roster (order matches the supplied legend, top to bottom) -----------
_HEX = Color.from_hex

INDIVIDUALS: tuple[Bird, ...] = (
    Bird("draft",  _HEX("#FFFFFF"), "steady",  "cockatiel"),
    Bird("pizza",  _HEX("#FF0000"), "flutter", "cockatiel"),
    Bird("percy",  _HEX("#FF9500"), "breathe", "lovebird"),
    Bird("matcha", _HEX("#00FF09"), "twinkle", "lovebird"),
    Bird("jynx",   _HEX("#0062FF"), "flutter", "lovebird"),
    Bird("bambi",  _HEX("#FFD000"), "bounce",  "budgie"),
)

# The 3 night/IR species outlines (what the night model emits). These drive the
# single "night LED" when the cameras are in IR.
NIGHT_SPECIES: tuple[Bird, ...] = (
    Bird("cockatiel", _HEX("#FFFFFF"), "breathe", "cockatiel", night=True),
    Bird("lovebird",  _HEX("#FF0000"), "breathe", "lovebird",  night=True),
    Bird("budgie",    _HEX("#FFD000"), "breathe", "budgie",    night=True),
)

ALL: tuple[Bird, ...] = INDIVIDUALS + NIGHT_SPECIES
_BY_NAME: dict[str, Bird] = {b.name: b for b in ALL}


def color_bgr(label: str, default: tuple[int, int, int] = (200, 200, 200)) -> tuple[int, int, int]:
    """A bird's signature roster color as an OpenCV BGR tuple (for drawing on frames).

    Falls back to a neutral gray for a label with no roster color (e.g.
    ``unknown_bird``). This is the SAME palette the LEDs use, so a bird's box and
    its light read as the same color.
    """
    bird = _BY_NAME.get(str(label).strip().lower())
    if bird is None:
        return default
    return (bird.color.b, bird.color.g, bird.color.r)

# Default left-to-right / top-to-bottom slot order for the 9-LED roster board.
ROSTER_ORDER: tuple[str, ...] = tuple(b.name for b in INDIVIDUALS) + tuple(b.name for b in NIGHT_SPECIES)
DAY_ORDER: tuple[str, ...] = tuple(b.name for b in INDIVIDUALS)
NIGHT_NAMES: frozenset[str] = frozenset(b.name for b in NIGHT_SPECIES)


def get(label: str) -> Bird | None:
    return _BY_NAME.get(label.lower())


def color_for(label: str, default: Color | None = None) -> Color:
    bird = _BY_NAME.get(label.lower())
    if bird is not None:
        return bird.color
    return default if default is not None else Color(40, 40, 40)


def is_night_species(label: str) -> bool:
    return label.lower() in NIGHT_NAMES


def is_individual(label: str) -> bool:
    b = _BY_NAME.get(label.lower())
    return b is not None and not b.night


def validate_against(species_members: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Return warnings if the LED roster has drifted from ``training/roster.yaml``.

    Only the *live* labels matter — archive-only deceased birds and the
    ``unknown_bird`` sentinel are never emitted by the live detector and are not
    expected to have LED colors, so they are excluded.
    """
    warnings: list[str] = []
    live = _load_live_labels()
    if live is None:
        # Couldn't read roster.yaml; do a forward-only check against whatever the
        # caller passed (so a typo'd LED bird name is still caught).
        if species_members is None:
            return warnings
        known = {m for members in species_members.values() for m in members} | set(species_members)
        for name in (*DAY_ORDER, *NIGHT_NAMES):
            if name not in known:
                warnings.append(f"LED roster lists '{name}' not present in roster.yaml")
        return warnings

    live_individuals, live_species = live
    for name in DAY_ORDER:
        if name not in live_individuals:
            warnings.append(f"LED roster lists individual '{name}' not in roster.yaml live model")
    for name in NIGHT_NAMES:
        if name not in live_species:
            warnings.append(f"LED roster lists night species '{name}' not in roster.yaml live model")
    for name in live_individuals:
        if name not in _BY_NAME:
            warnings.append(f"live individual '{name}' has no LED color (add to rgb/birds.py)")
    for name in live_species:
        if name not in _BY_NAME:
            warnings.append(f"live species '{name}' has no LED color (add to rgb/birds.py)")
    return warnings


def _load_live_labels() -> tuple[set[str], set[str]] | None:
    """Read roster.yaml and return (live individuals, live species) lowercased.

    Mirrors lib.roster's search paths: cwd-relative first, then repo-anchored.
    Returns None if the file can't be read.
    """
    from pathlib import Path

    candidates = (
        Path("training/roster.yaml"),
        Path(__file__).resolve().parents[3] / "training" / "roster.yaml",
    )
    for path in candidates:
        try:
            if not path.exists():
                continue
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        individuals: set[str] = set()
        species: set[str] = set()
        for entry in data.get("labels", []):
            models = entry.get("models", []) or []
            if "live" not in models:
                continue
            name = str(entry.get("name", "")).lower()
            kind = entry.get("kind")
            if kind == "individual":
                individuals.add(name)
            elif kind == "species":
                species.add(name)
        return individuals, species
    return None
