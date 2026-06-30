from __future__ import annotations

from datetime import date, datetime, time

from lib.suntimes import sun_times


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


# Location is passed explicitly (no real deployment coordinates live in the repo).
# A generic tropical site at UTC+8 gives a clear morning sunrise / evening sunset.
_LAT, _LON, _OFF = 14.0, 121.0, 8.0


def test_tropical_midsummer_sunrise_and_sunset() -> None:
    sunrise, sunset = sun_times(date(2026, 6, 25), _LAT, _LON, _OFF)
    assert sunrise is not None and sunset is not None
    # ~14°N at UTC+8 (longitude 121): sunrise early morning, sunset early evening.
    assert time(5, 0) <= sunrise <= time(6, 0)
    assert time(18, 0) <= sunset <= time(19, 0)
    assert sunrise < sunset


def test_equinox_day_length_is_about_twelve_hours() -> None:
    sunrise, sunset = sun_times(date(2026, 3, 20), _LAT, _LON, _OFF)
    assert sunrise is not None and sunset is not None
    day_minutes = _minutes(sunset) - _minutes(sunrise)
    assert abs(day_minutes - 12 * 60) <= 30


def test_winter_day_is_shorter_than_summer_day() -> None:
    sr_w, ss_w = sun_times(date(2026, 12, 21), _LAT, _LON, _OFF)
    sr_s, ss_s = sun_times(date(2026, 6, 21), _LAT, _LON, _OFF)
    winter = _minutes(ss_w) - _minutes(sr_w)
    summer = _minutes(ss_s) - _minutes(sr_s)
    assert summer > winter  # tropics: small but real seasonal swing
