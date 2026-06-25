"""Pure sleep DETECTION: a reducer over a stream of light/motion/clock events.

The flock's night is read from the same IR signal the care scheduler trusts: the
room going fully dark (``all_ir`` and at least one camera) starts a night; the
room lightening for good in the morning ends it. In between, mid-night re-lights
and night-motion bursts are recorded as discrete, debounced disturbances.

Everything is a pure ``step(state, event) -> state`` over ``Ir`` / ``Motion`` /
``Tick`` events, so it unit-tests with synthetic event streams and no threads,
clock, or cameras. Detection only fills the night's TIMING + disturbances; the
0-100 score is applied separately (by the tracker, via :mod:`lib.sleep.score`).
All thresholds are named, tunable constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from lib.sleep.model import LIGHT, MOTION, NIGHT_FRIGHT, Disturbance, SleepNight

# Debounce / confirmation windows (minutes).
DARK_CONFIRM_MIN = 5      # sustained dark before a night opens (reject evening flicker)
WAKE_CONFIRM_MIN = 10     # sustained morning light before a night ends
RELIGHT_GRACE_MIN = 8     # a re-light returning to dark within this is one brief event
MOTION_DEBOUNCE_MIN = 15  # collapse a sustained thrash into one disturbance
MIN_NIGHT_MIN = 120       # a night must be this old before a leave-IR can end it

# Motion thresholds (movement_percent from the registry).
FRIGHT_MOVE = 35.0        # a panicked thrash in the dark
MOTION_MOVE = 12.0        # a lesser night stir

# Time-of-day guards.
EVENING_HOUR = 16         # a night can only open at >= 16:00 or <= 04:00
MORNING_LO = time(4, 0)   # a wake (leave-IR end) must fall in [04:00, 11:00]
MORNING_HI = time(11, 0)
RECENT_LIGHT_MIN = 3      # motion within this of a leave-IR is a human/light, not a fright

# Forced-finalize backstops (a camera stuck IR mid-night never re-lightens).
MAX_NIGHT_HOURS = 16
FORCE_FINALIZE_HOUR = 14


@dataclass(frozen=True)
class Ir:
    """The room's dark state recomputed after an IR transition."""

    at: datetime
    dark: bool       # all_ir() AND camera_count > 0
    cameras: int


@dataclass(frozen=True)
class Motion:
    at: datetime
    movement_percent: float


@dataclass(frozen=True)
class Tick:
    at: datetime


@dataclass
class NightState:
    """Open night (or None when awake) + debounce bookkeeping. Driven serially."""

    night: SleepNight | None = None
    just_finalized: SleepNight | None = None  # set on the step that ends a night
    _dark_since: datetime | None = None
    _dark_cameras: int = 0
    _leave_since: datetime | None = None
    _last_leave_at: datetime | None = None
    _relit_minutes: float = 0.0
    _last_motion: datetime | None = None
    _last_cameras: int = 0


def _mins(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60.0


def night_of(when: datetime) -> date:
    """The date a night BELONGS to (its evening). A night confirmed after
    midnight (hour < 12) files under the previous calendar date."""
    return when.date() if when.hour >= 12 else when.date() - timedelta(days=1)


def should_open(now: datetime) -> bool:
    """Evening guard: a night may only open in the evening or pre-dawn, so a
    stray midday all_ir (a storm, curtains) can't start one."""
    return now.time() >= time(EVENING_HOUR, 0) or now.time() <= time(4, 0)


def _is_morning(when: datetime) -> bool:
    return MORNING_LO <= when.time() <= MORNING_HI


def start_night(lights_out: datetime, cameras: int) -> SleepNight:
    return SleepNight(night_of=night_of(lights_out), lights_out=lights_out, camera_count_at_dark=cameras)


def _recent_light(state: NightState, now: datetime) -> bool:
    if state._leave_since is not None:
        return True
    return state._last_leave_at is not None and _mins(state._last_leave_at, now) <= RECENT_LIGHT_MIN


def classify_motion(movement: float, state: NightState, now: datetime) -> Disturbance | None:
    """A night-motion event into a fright/motion disturbance, or None.

    A spike WITH a concurrent/recent light is a human/light (handled as a light
    event), never a fright. Debounced so a sustained thrash counts once.
    """
    if movement < MOTION_MOVE or _recent_light(state, now):
        return None
    if state._last_motion is not None and _mins(state._last_motion, now) < MOTION_DEBOUNCE_MIN:
        return None
    if movement >= FRIGHT_MOVE:
        return Disturbance(now, NIGHT_FRIGHT, detail=f"possible night-fright ~{now.strftime('%H:%M')}")
    return Disturbance(now, MOTION, detail=f"a stir ~{now.strftime('%H:%M')}")


def _finalize(state: NightState, first_light: datetime, *, partial: bool, note: str | None = None) -> None:
    night = state.night
    assert night is not None
    # Close any still-open lit interval (a light left on through finalize) so the
    # force-finalize path excludes it from the dark window — otherwise a left-on
    # light would be counted as dark. A no-op for the morning-wake path, which
    # finalizes exactly at _leave_since.
    if state._leave_since is not None and first_light > state._leave_since:
        state._relit_minutes += _mins(state._leave_since, first_light)
    gross = _mins(night.lights_out, first_light) if night.lights_out else 0.0
    night.first_light = first_light
    night.dark_minutes = max(0, round(gross - state._relit_minutes))
    night.camera_count_at_wake = state._last_cameras
    if partial:
        night.partial_coverage = True
    if note:
        night.notes.append(note)
    night.finalized = True
    state.just_finalized = night
    # Back to AWAKE; clear all bookkeeping.
    state.night = None
    state._dark_since = None
    state._leave_since = None
    state._last_leave_at = None
    state._relit_minutes = 0.0
    state._last_motion = None


def step(state: NightState, event) -> NightState:
    """Advance the detection state by one event (mutates + returns ``state``)."""
    state.just_finalized = None
    now = event.at

    if isinstance(event, Ir):
        state._last_cameras = event.cameras
        if state.night is None:
            # AWAKE: arm/disarm the dark-confirmation timer.
            if event.dark:
                if state._dark_since is None:
                    state._dark_since = now
                    state._dark_cameras = event.cameras
            else:
                state._dark_since = None  # evening flicker before bedtime
        else:
            # DARK (night open).
            if event.cameras != state.night.camera_count_at_dark:
                state.night.partial_coverage = True
            if event.dark:
                # Back to dark — close an open lit interval as ONE light event.
                if state._leave_since is not None:
                    dur = _mins(state._leave_since, now)
                    state.night.disturbances.append(
                        Disturbance(state._leave_since, LIGHT, detail=f"a light ~{state._leave_since.strftime('%H:%M')}", minutes=round(dur, 1))
                    )
                    state._relit_minutes += dur
                    state._leave_since = None
            else:
                # Room lit: arm the leave timer (could be a flicker, a sustained
                # light, or the morning wake — resolved on later events).
                if state._leave_since is None:
                    state._leave_since = now
                    state._last_leave_at = now

    elif isinstance(event, Motion):
        if state.night is not None:
            state.night.max_motion = max(state.night.max_motion, event.movement_percent)
            disturbance = classify_motion(event.movement_percent, state, now)
            if disturbance is not None:
                state.night.disturbances.append(disturbance)
                state._last_motion = now

    # Every event also advances the clock — fold in the time-based transitions.
    _advance(state, now)
    return state


def _advance(state: NightState, now: datetime) -> None:
    """Time-driven transitions: confirm a dark start, a morning wake, or force-end."""
    if state.night is None:
        # Guard on when the dark ACTUALLY began, not the confirm instant — a real
        # bedtime at 03:57 that confirms at 04:02 must still open.
        if (
            state._dark_since is not None
            and _mins(state._dark_since, now) >= DARK_CONFIRM_MIN
            and should_open(state._dark_since)
        ):
            state.night = start_night(state._dark_since, state._dark_cameras)
            state._dark_since = None
        return

    # Night open: a sustained MORNING leave-IR is the wake (finalize).
    if (
        state._leave_since is not None
        and _is_morning(state._leave_since)
        and _mins(state._leave_since, now) >= WAKE_CONFIRM_MIN
        and _mins(state.night.lights_out, now) >= MIN_NIGHT_MIN
    ):
        _finalize(state, state._leave_since, partial=False)
        return

    # Backstop: a camera stuck IR (or a light left on all night) never gives a
    # clean wake — force-finalize with reduced confidence. Either the night has
    # run absurdly long, or it's the NEXT afternoon (past 14:00 on a later date)
    # with still no wake. The next-day guard is essential: a normal evening time
    # like 23:00 is >= 14:00 too and must NOT trip this.
    age_h = _mins(state.night.lights_out, now) / 60.0
    # Key the "next afternoon" backstop off the night's evening date so a pre-dawn
    # start (lights_out after midnight) still resolves on the right calendar day.
    next_afternoon = now.date() > state.night.night_of and now.time() >= time(FORCE_FINALIZE_HOUR, 0)
    if age_h > MAX_NIGHT_HOURS or next_afternoon:
        _finalize(state, now, partial=True, note="no clear wake observed — estimated")
