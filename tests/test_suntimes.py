from __future__ import annotations

from datetime import date, datetime, time

from lib.suntimes import sun_times


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def test_manila_midsummer_sunrise_and_sunset() -> None:
    sunrise, sunset = sun_times(date(2026, 6, 25))
    assert sunrise is not None and sunset is not None
    # Real Manila almanac is ~05:28 / ~18:27; allow a few minutes of slack.
    assert time(5, 15) <= sunrise <= time(5, 45)
    assert time(18, 15) <= sunset <= time(18, 45)
    assert sunrise < sunset


def test_equinox_day_length_is_about_twelve_hours() -> None:
    sunrise, sunset = sun_times(date(2026, 3, 20))
    assert sunrise is not None and sunset is not None
    day_minutes = _minutes(sunset) - _minutes(sunrise)
    assert abs(day_minutes - 12 * 60) <= 30


def test_winter_day_is_shorter_than_summer_day() -> None:
    sr_w, ss_w = sun_times(date(2026, 12, 21))
    sr_s, ss_s = sun_times(date(2026, 6, 21))
    winter = _minutes(ss_w) - _minutes(sr_w)
    summer = _minutes(ss_s) - _minutes(sr_s)
    assert summer > winter  # tropics: small but real seasonal swing
