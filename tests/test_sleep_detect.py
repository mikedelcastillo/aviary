from __future__ import annotations

from datetime import date, datetime

from lib.sleep.detect import Ir, Motion, NightState, Tick, classify_motion, night_of, should_open, step
from lib.sleep.model import LIGHT, MOTION, NIGHT_FRIGHT


def run(events) -> NightState:
    state = NightState()
    for event in events:
        step(state, event)
    return state


def ev(day, h, m):
    return datetime(2026, 6, day, h, m)


# -- opening a night ---------------------------------------------------------


def test_evening_flicker_does_not_open() -> None:
    state = run([Ir(ev(25, 20, 0), True, 2), Ir(ev(25, 20, 3), False, 2)])  # < 5 min dark
    assert state.night is None


def test_sustained_evening_dark_opens_with_lights_out_at_run_start() -> None:
    state = run([Ir(ev(25, 20, 0), True, 2), Tick(ev(25, 20, 6))])
    assert state.night is not None
    assert state.night.lights_out == ev(25, 20, 0)  # start of the confirmed run, not the confirm instant
    assert state.night.camera_count_at_dark == 2
    assert state.night.night_of == date(2026, 6, 25)


def test_midday_dark_does_not_open() -> None:
    state = run([Ir(ev(25, 12, 0), True, 2), Tick(ev(25, 12, 6))])
    assert state.night is None  # evening guard rejects a midday storm/curtains


def test_post_midnight_open_files_under_previous_date() -> None:
    state = run([Ir(ev(26, 0, 24), True, 2), Tick(ev(26, 0, 30))])
    assert state.night is not None
    assert state.night.night_of == date(2026, 6, 25)  # the evening it belongs to


# -- light disturbances ------------------------------------------------------


def _opened():
    state = run([Ir(ev(25, 20, 0), True, 2), Tick(ev(25, 20, 6))])
    assert state.night is not None
    return state


def test_relight_roundtrip_is_one_light_event_excluded_from_dark() -> None:
    state = _opened()
    step(state, Ir(ev(25, 23, 0), False, 2))  # room lit
    step(state, Ir(ev(25, 23, 5), True, 2))   # back to dark within grace
    lights = [d for d in state.night.disturbances if d.kind == LIGHT]
    assert len(lights) == 1 and lights[0].minutes == 5.0
    # The 5 lit minutes are excluded from the eventual dark window.
    step(state, Ir(ev(26, 7, 0), False, 2))
    step(state, Tick(ev(26, 7, 11)))
    assert state.just_finalized is not None
    gross = (ev(26, 7, 0) - ev(25, 20, 0)).total_seconds() / 60
    assert state.just_finalized.dark_minutes == round(gross - 5)


def test_sustained_light_records_its_duration() -> None:
    state = _opened()
    step(state, Ir(ev(25, 23, 0), False, 2))
    step(state, Ir(ev(25, 23, 50), True, 2))  # lit for 50 min then dark
    lights = [d for d in state.night.disturbances if d.kind == LIGHT]
    assert len(lights) == 1 and lights[0].minutes == 50.0


# -- motion / night-frights --------------------------------------------------


def test_motion_classification() -> None:
    state = _opened()
    assert classify_motion(40.0, state, ev(26, 2, 0)).kind == NIGHT_FRIGHT
    assert classify_motion(20.0, state, ev(26, 2, 0)).kind == MOTION
    assert classify_motion(8.0, state, ev(26, 2, 0)) is None


def test_motion_with_concurrent_light_is_not_a_fright() -> None:
    state = _opened()
    step(state, Ir(ev(26, 2, 0), False, 2))  # a light is on
    step(state, Motion(ev(26, 2, 1), 40.0))  # spike WITH the light -> human/light
    assert not any(d.kind == NIGHT_FRIGHT for d in state.night.disturbances)


def test_fright_is_debounced() -> None:
    state = _opened()
    step(state, Motion(ev(26, 2, 0), 40.0))
    step(state, Motion(ev(26, 2, 10), 40.0))  # within 15-min debounce
    assert len([d for d in state.night.disturbances if d.kind == NIGHT_FRIGHT]) == 1


# -- finalize ----------------------------------------------------------------


def test_sustained_morning_leave_finalizes() -> None:
    state = _opened()
    step(state, Ir(ev(26, 6, 0), False, 2))   # leave IR in the morning
    step(state, Tick(ev(26, 6, 11)))          # sustained > 10 min
    night = state.just_finalized
    assert night is not None and night.finalized
    assert night.first_light == ev(26, 6, 0)
    assert night.dark_minutes == 600  # 20:00 -> 06:00


def test_morning_leave_before_min_night_does_not_finalize() -> None:
    # Open at 03:00 (pre-dawn guard), a 04:30 leave is only ~1.5h in -> not a wake.
    state = run([Ir(ev(26, 3, 0), True, 2), Tick(ev(26, 3, 6))])
    assert state.night is not None
    step(state, Ir(ev(26, 4, 30), False, 2))
    step(state, Tick(ev(26, 4, 41)))
    assert state.just_finalized is None and state.night is not None


def test_force_finalize_when_no_wake_observed() -> None:
    state = _opened()
    step(state, Tick(ev(26, 14, 0)))  # 18h in / past 14:00 with no wake
    night = state.just_finalized
    assert night is not None and night.finalized and night.partial_coverage


def test_force_finalize_excludes_a_left_on_light_from_dark() -> None:
    state = _opened()  # lights_out 20:00
    step(state, Ir(ev(25, 23, 0), False, 2))  # a light comes on and never returns to dark
    step(state, Tick(ev(26, 14, 0)))          # force-finalize the next afternoon
    night = state.just_finalized
    assert night is not None and night.partial_coverage
    assert night.dark_minutes == 180  # only 20:00->23:00 was truly dark, not the lit 15h


def test_pre_dawn_dark_confirming_after_4am_opens() -> None:
    # A real bedtime at 03:57 confirmed at 04:02 must still open (guard checks the
    # dark-start instant, not the confirm instant).
    state = run([Ir(ev(26, 3, 57), True, 2), Tick(ev(26, 4, 2))])
    assert state.night is not None
    assert state.night.lights_out == ev(26, 3, 57)


def test_helpers() -> None:
    assert night_of(datetime(2026, 6, 26, 0, 30)) == date(2026, 6, 25)
    assert night_of(datetime(2026, 6, 25, 20, 0)) == date(2026, 6, 25)
    assert should_open(datetime(2026, 6, 25, 20, 0)) is True
    assert should_open(datetime(2026, 6, 25, 12, 0)) is False
