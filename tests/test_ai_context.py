from __future__ import annotations

from datetime import datetime

from lib.ai.context import build_chat_context, format_system_state

MEMBER_SPECIES = {"percy": "lovebird", "draft": "cockatiel", "bambi": "budgie"}
NOON = datetime(2026, 6, 25, 12, 30)


def test_system_state_reports_time_health_and_visibility() -> None:
    state = format_system_state(
        NOON,
        cameras_total=3,
        cameras_healthy=2,
        visible_text="Percy (Big Cage)",
        daylight="day",
        autofind_on=True,
    )
    assert "12:30" in state
    assert "2 of 3" in state
    assert "Percy (Big Cage)" in state
    assert "daylight" in state
    assert "Auto-find" in state and "ON" in state


def test_system_state_paused_hides_sightings() -> None:
    state = format_system_state(
        NOON,
        paused=True,
        pause_status="Privacy mode on — cameras paused for 10m.",
        cameras_total=2,
        cameras_healthy=0,
        visible_text="Percy (Big Cage)",  # must be ignored while paused
    )
    assert "paused for 10m" in state  # the pause status is surfaced verbatim
    assert "Big Cage" not in state  # no sightings claimed while paused
    assert "daylight" not in state  # no daylight claim while paused


def test_system_state_no_cameras_makes_no_daylight_claim() -> None:
    # With no cameras online, don't assert "the cameras are in daylight".
    state = format_system_state(NOON, cameras_total=0, cameras_healthy=0)
    assert "No cameras are online" in state
    assert "daylight" not in state and "night/IR" not in state


def test_system_state_night_mode() -> None:
    state = format_system_state(NOON, cameras_total=2, cameras_healthy=2, daylight="night")
    assert "night/IR" in state and "roosting" in state.lower()


def test_build_chat_context_combines_state_and_care() -> None:
    state = format_system_state(NOON, cameras_total=1, cameras_healthy=1)
    context = build_chat_context(
        "can the birds eat avocado?", system_state=state, member_species=MEMBER_SPECIES
    )
    assert context is not None
    assert "Current aviary state" in context  # state block present
    assert "avocado" in context.lower() and "TOXIC" in context  # care block present


def test_build_chat_context_state_only_for_status_question() -> None:
    state = format_system_state(NOON, cameras_total=1, cameras_healthy=1)
    context = build_chat_context("how are the cameras?", system_state=state)
    assert context is not None
    assert "Current aviary state" in context


def test_build_chat_context_none_when_nothing_relevant() -> None:
    # No live state and pure chit-chat -> nothing to ground.
    assert build_chat_context("hello there!", system_state=None) is None
