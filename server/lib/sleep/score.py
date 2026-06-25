"""The nightly bird sleep score (pure, research-grounded).

A 0-100 score from four components, weighted the way the sleep research weights
them — dark DURATION dominates, schedule CONSISTENCY is nearly as important, and
light-at-night + disturbances are the two penalising disruptors:

    DURATION 0.40 · CONSISTENCY 0.25 · DARKNESS 0.20 · DISTURBANCE 0.15

Targets come from the verified care research (``lib.care.SLEEP_HOURS`` = 10-12h
of dark; a consistent bedtime within ~30 min; night-frights are the heaviest
single disturbance). ``confidence`` is reported SEPARATELY from the number — it
only hedges the copy, it never inflates or deflates the score.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.care import SLEEP_HOURS
from lib.sleep.model import LIGHT, MOTION, NIGHT_FRIGHT, SleepNight

WEIGHTS = {"duration": 0.40, "consistency": 0.25, "darkness": 0.20, "disturbance": 0.15}

TARGET_LOW, TARGET_HIGH = SLEEP_HOURS  # (10, 12)
# Consistency defaults to neutral until we have a baseline, so a cold start isn't
# punished for having no history.
NEUTRAL_CONSISTENCY = 0.7


@dataclass(frozen=True)
class Baseline:
    """Rolling median bedtime/wake (minutes-of-day) from recent nights."""

    lights_out_min: float
    first_light_min: float


def duration_score(hours: float) -> float:
    """Dark-hours -> 0..1. Full credit in the 10-12h band, steep penalty under 8,
    no reward above target (over-long isn't harmful but isn't rewarded)."""
    if hours >= TARGET_LOW:
        return 1.0 if hours <= 14 else 0.9
    if hours >= 8:
        return 0.6 + 0.4 * (hours - 8) / 2.0
    if hours >= 6:
        return 0.2 + 0.4 * (hours - 6) / 2.0
    if hours > 0:
        return 0.2 * (hours / 6.0)
    return 0.0


def _deviation_score(minutes: float) -> float:
    """How close a clock-time is to its baseline: <=30 min perfect, >120 min zero."""
    if minutes <= 30:
        return 1.0
    if minutes <= 60:
        return 1.0 - 0.4 * (minutes - 30) / 30.0
    if minutes <= 120:
        return 0.6 - 0.4 * (minutes - 60) / 60.0
    return 0.0


def _minutes_of_day(dt) -> int:
    return dt.hour * 60 + dt.minute


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def rolling_baseline(prior: list[SleepNight]) -> Baseline | None:
    """Median bedtime/wake over the most recent prior finalized nights (<=7).

    Returns None with fewer than 2 usable nights — the caller then uses a neutral
    consistency so a cold start isn't penalised. Median (not mean) so a single
    odd night doesn't poison the baseline.
    """
    recent = [n for n in prior if n.finalized][-7:]
    outs = [_minutes_of_day(n.lights_out) for n in recent if n.lights_out is not None]
    lights = [_minutes_of_day(n.first_light) for n in recent if n.first_light is not None]
    if len(outs) < 2 or len(lights) < 2:
        return None
    return Baseline(_median(outs), _median(lights))


def consistency_score(night: SleepNight, baseline: Baseline | None) -> float:
    if baseline is None or night.lights_out is None or night.first_light is None:
        return NEUTRAL_CONSISTENCY
    out_dev = abs(_minutes_of_day(night.lights_out) - baseline.lights_out_min)
    light_dev = abs(_minutes_of_day(night.first_light) - baseline.first_light_min)
    return (_deviation_score(out_dev) + _deviation_score(light_dev)) / 2.0


def darkness_score(disturbances) -> float:
    """1.0 minus per-LIGHT-event penalties, scaled by how long the room was lit.

    A dim steady protective night-light keeps cameras in IR and emits no light
    event, so it's never penalised — only a real camera leaving IR (room/TV
    light) costs points. That falls out of the signal for free.
    """
    penalty = 0.0
    for d in disturbances:
        if d.kind != LIGHT:
            continue
        if d.minutes > 30:
            penalty += 0.40
        elif d.minutes <= 8:
            penalty += 0.15
        else:
            penalty += 0.25
    return max(0.0, 1.0 - penalty)


def disturbance_score(disturbances) -> float:
    """1.0 minus motion/fright penalties. Light events are NOT re-counted here
    (they're in :func:`darkness_score`), so there's no double-penalty."""
    penalty = 0.0
    for d in disturbances:
        if d.kind == NIGHT_FRIGHT:
            penalty += 0.40
        elif d.kind == MOTION:
            penalty += 0.10
    return max(0.0, 1.0 - penalty)


def _confidence(night: SleepNight, baseline: Baseline | None) -> float:
    factor = 1.0
    if night.partial_coverage:
        factor *= 0.6
    if night.camera_count_at_dark == 1:
        factor *= 0.8
    if baseline is None:
        factor *= 0.85
    if night.camera_count_at_wake and night.camera_count_at_wake != night.camera_count_at_dark:
        factor *= 0.8
    return max(0.3, min(1.0, factor))


def score_night(night: SleepNight, baseline: Baseline | None) -> tuple[int, dict[str, float], float]:
    """Return ``(score 0-100, component sub-scores 0-1, confidence 0-1)``."""
    components = {
        "duration": round(duration_score(night.dark_hours), 4),
        "consistency": round(consistency_score(night, baseline), 4),
        "darkness": round(darkness_score(night.disturbances), 4),
        "disturbance": round(disturbance_score(night.disturbances), 4),
    }
    raw = sum(WEIGHTS[key] * value for key, value in components.items())
    return round(100 * raw), components, _confidence(night, baseline)
