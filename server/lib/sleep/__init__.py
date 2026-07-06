"""Flock-level bird sleep tracking: nightly sleep scores + a warm summary.

The room darkens (all cameras in IR) and lightens together, so this measures the
FLOCK's sleep honestly from that shared IR signal — no fabricated per-bird
precision. ``model`` holds the records + persistence;
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
from lib.sleep.narrate import (
    COLD_START_NOTE,
    NO_COVERAGE,
    format_last,
    format_morning,
    format_status_line,
    format_week,
    sleep_streak,
)
from lib.sleep.tracker import SleepTracker

__all__ = [
    "LIGHT",
    "MOTION",
    "NIGHT_FRIGHT",
    "COLD_START_NOTE",
    "NO_COVERAGE",
    "Disturbance",
    "SleepNight",
    "SleepTracker",
    "append_night",
    "clear_current",
    "format_last",
    "format_morning",
    "format_status_line",
    "format_week",
    "sleep_streak",
    "from_json",
    "load_current",
    "load_nights",
    "load_recent",
    "save_current",
    "to_json",
]
