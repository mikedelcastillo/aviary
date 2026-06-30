"""Local sunrise/sunset times, with no external dependency.

The care scheduler anchors the day to the sun — a wind-down before sunset, a
morning routine near sunrise — so the birds' light/dark cycle stays natural and
consistent. This implements the standard "sunrise equation" (NOAA/Almanac) and
returns naive local times for a fixed UTC offset.

The location and UTC offset are supplied by the caller from configuration (set in
.env), never hardcoded here — the defaults below are a neutral 0,0/UTC placeholder
so no real location lives in the repo. Returned times are tz-naive, matching
:mod:`lib.clock`.
"""

from __future__ import annotations

import math
from datetime import date, time

# Neutral placeholders — the real location + UTC offset come from config (.env).
DEFAULT_LATITUDE = 0.0
DEFAULT_LONGITUDE = 0.0
DEFAULT_UTC_OFFSET_HOURS = 0.0

# Official sunrise/sunset zenith (accounts for atmospheric refraction + sun disc).
_ZENITH = 90.833


def _event(day: date, latitude: float, longitude: float, tz_offset: float, *, sunrise: bool) -> time | None:
    """One solar event (sunrise or sunset) as a local time, or None if it never
    happens that day (polar latitudes — never the case for the tropics)."""
    n = day.timetuple().tm_yday
    lng_hour = longitude / 15.0

    t = n + ((6.0 if sunrise else 18.0) - lng_hour) / 24.0
    # Sun's mean anomaly, true longitude.
    m = (0.9856 * t) - 3.289
    L = m + (1.916 * math.sin(math.radians(m))) + (0.020 * math.sin(math.radians(2 * m))) + 282.634
    L %= 360.0

    # Right ascension, put in the same quadrant as L.
    ra = math.degrees(math.atan(0.91764 * math.tan(math.radians(L)))) % 360.0
    ra += (math.floor(L / 90.0) * 90.0) - (math.floor(ra / 90.0) * 90.0)
    ra /= 15.0

    sin_dec = 0.39782 * math.sin(math.radians(L))
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = (math.cos(math.radians(_ZENITH)) - (sin_dec * math.sin(math.radians(latitude)))) / (
        cos_dec * math.cos(math.radians(latitude))
    )
    if cos_h > 1.0 or cos_h < -1.0:
        return None  # sun never rises / never sets at this latitude today

    h = (360.0 - math.degrees(math.acos(cos_h))) if sunrise else math.degrees(math.acos(cos_h))
    h /= 15.0

    local = (h + ra - (0.06571 * t) - 6.622 - lng_hour + tz_offset) % 24.0
    hour = int(local)
    minute = int(round((local - hour) * 60.0))
    if minute == 60:
        hour = (hour + 1) % 24
        minute = 0
    return time(hour, minute)


def sun_times(
    day: date,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    tz_offset_hours: float = DEFAULT_UTC_OFFSET_HOURS,
) -> tuple[time | None, time | None]:
    """Return ``(sunrise, sunset)`` local times for ``day`` at the location."""
    return (
        _event(day, latitude, longitude, tz_offset_hours, sunrise=True),
        _event(day, latitude, longitude, tz_offset_hours, sunrise=False),
    )
