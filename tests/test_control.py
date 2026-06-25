from __future__ import annotations

from lib.control import RuntimeControl, parse_duration


# -- parse_duration ---------------------------------------------------------


def test_parse_duration_bare_number_is_minutes() -> None:
    assert parse_duration("5") == 5 * 60


def test_parse_duration_unit_abbreviations() -> None:
    assert parse_duration("1m") == 60
    assert parse_duration("1h") == 3600
    assert parse_duration("30s") == 30
    assert parse_duration("2d") == 2 * 86400


def test_parse_duration_spelled_out_with_space() -> None:
    assert parse_duration("1 min") == 60
    assert parse_duration("1 minute") == 60
    assert parse_duration("2 hours") == 7200
    assert parse_duration("45 seconds") == 45


def test_parse_duration_combined_clauses() -> None:
    assert parse_duration("1h30m") == 3600 + 1800
    assert parse_duration("1h 30m") == 3600 + 1800


def test_parse_duration_indefinite_inputs_return_none() -> None:
    assert parse_duration("") is None
    assert parse_duration(None) is None
    assert parse_duration("   ") is None
    assert parse_duration("forever") is None
    assert parse_duration("0") is None


# -- RuntimeControl ---------------------------------------------------------


def test_starts_live() -> None:
    control = RuntimeControl()
    assert control.is_paused() is False
    assert "live" in control.status().lower()


def test_indefinite_pause_stays_paused() -> None:
    now = 100.0
    control = RuntimeControl(clock=lambda: now)
    control.pause(None)
    assert control.is_paused() is True
    assert control.remaining_seconds() is None
    now = 100_000.0
    # No deadline -> never lapses on its own.
    assert control.is_paused() is True


def test_timed_pause_auto_resumes_after_deadline() -> None:
    now = 0.0
    control = RuntimeControl(clock=lambda: now)
    control.pause(60)
    assert control.is_paused() is True
    assert control.remaining_seconds() == 60
    now = 59.0
    assert control.is_paused() is True
    now = 60.0
    # Deadline reached: is_paused self-clears and the state goes back to live.
    assert control.is_paused() is False
    assert control.remaining_seconds() is None
    assert "live" in control.status().lower()


def test_resume_clears_pause() -> None:
    control = RuntimeControl()
    control.pause(None)
    message = control.resume()
    assert control.is_paused() is False
    assert "resumed" in message.lower()


def test_resume_when_already_live_is_idempotent() -> None:
    control = RuntimeControl()
    message = control.resume()
    assert control.is_paused() is False
    assert "already" in message.lower()


def test_status_reports_remaining_for_timed_pause() -> None:
    now = 0.0
    control = RuntimeControl(clock=lambda: now)
    control.pause(125)
    status = control.status()
    assert "2m" in status
    assert "paused" in status.lower()
