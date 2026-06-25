"""Flock-level bird sleep tracking: nightly sleep scores + a warm summary.

The room darkens (all cameras in IR) and lightens together, so this measures the
FLOCK's sleep honestly from the same IR signal the care scheduler trusts — no
fabricated per-bird precision. ``model`` holds the records + persistence;
``score`` the research-grounded 0-100 score; ``detect`` the pure detection state
machine; ``narrate`` the warm copy; ``tracker`` the one thin threaded shell.
"""

from __future__ import annotations

from lib.sleep.model import (
    LIGHT,
    MOTION,
    NIGHT_FRIGHT,
    Disturbance,
    SleepNight,
    append_night,
    clear_current,
    from_json,
    load_current,
    load_nights,
    load_recent,
    save_current,
    to_json,
)

__all__ = [
    "LIGHT",
    "MOTION",
    "NIGHT_FRIGHT",
    "Disturbance",
    "SleepNight",
    "append_night",
    "clear_current",
    "from_json",
    "load_current",
    "load_nights",
    "load_recent",
    "save_current",
    "to_json",
]
