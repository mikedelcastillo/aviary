"""Weather outlook + bird-safety advice for the flock.

Pulls a short forecast for the configured location from Open-Meteo (a free,
keyless API — only latitude/longitude are sent, and those live in ``.env``, never
in the repo) and turns it into:

  * an on-demand ``/weather`` summary (today's high/low, conditions, and concrete
    advice for the birds), and
  * proactive warnings, checked several times a day, when the day will be too hot
    or the night too cold for tropical cage birds.

Cockatiels, lovebirds and budgies are comfortable roughly 18–29 °C (65–85 °F).
Sustained heat above ~32 °C risks heat stress; nights below ~10 °C are too cold
for them and below ~4 °C is dangerous — hence the default thresholds.

The parsing/assessment/formatting are pure functions so they unit-test without a
network call; :func:`fetch_forecast` is the only part that talks to the API.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import requests

from lib.clock import now_ph

LOGGER = logging.getLogger("lib.weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Bird-safety temperature thresholds in °C (overridable via env / config).
DEFAULT_HOT_C = 32.0
DEFAULT_COLD_C = 10.0

# Checked this often through the day; per-day dedup keeps each warning to once.
DEFAULT_POLL_SECONDS = 6 * 3600.0

# A compact subset of WMO weather codes -> human text.
_WMO = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def _describe_code(code: int | None) -> str:
    if code is None:
        return "unknown conditions"
    return _WMO.get(int(code), "mixed conditions")


@dataclass(frozen=True)
class WeatherForecast:
    date_label: str
    high_c: float
    low_c: float
    condition: str
    current_c: float | None = None
    humidity: int | None = None
    precip_prob: int | None = None


@dataclass(frozen=True)
class WeatherAssessment:
    is_hot: bool
    is_cold: bool
    is_wet: bool
    warnings: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)


def has_location(latitude: float, longitude: float) -> bool:
    """True if a real location is configured (not the 0,0 placeholder default)."""
    return not (abs(latitude) < 1e-6 and abs(longitude) < 1e-6)


def _first(seq: object, default: object = None) -> object:
    if isinstance(seq, list) and seq:
        return seq[0]
    return default


def parse_forecast(payload: dict) -> WeatherForecast | None:
    """Turn an Open-Meteo response into a :class:`WeatherForecast` (None if unusable)."""
    try:
        daily = payload["daily"]
        high = float(_first(daily["temperature_2m_max"]))
        low = float(_first(daily["temperature_2m_min"]))
    except (KeyError, TypeError, ValueError):
        LOGGER.warning("Weather response missing daily temperatures: %r", str(payload)[:200])
        return None

    date_label = str(_first(daily.get("time"), "") or "")
    code = _first(daily.get("weather_code"))
    precip = _first(daily.get("precipitation_probability_max"))
    current = payload.get("current") or {}
    current_c = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")

    return WeatherForecast(
        date_label=date_label,
        high_c=high,
        low_c=low,
        condition=_describe_code(code),
        current_c=float(current_c) if isinstance(current_c, (int, float)) else None,
        humidity=int(humidity) if isinstance(humidity, (int, float)) else None,
        precip_prob=int(precip) if isinstance(precip, (int, float)) else None,
    )


def fetch_forecast(
    latitude: float,
    longitude: float,
    *,
    timeout: float = 10.0,
) -> WeatherForecast | None:
    """Fetch today's forecast from Open-Meteo. Returns None on any failure."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        response.raise_for_status()
        return parse_forecast(response.json())
    except (requests.RequestException, ValueError):
        LOGGER.exception("Weather fetch failed")
        return None


def _c_to_f(celsius: float) -> int:
    return round(celsius * 9 / 5 + 32)


def _temp(celsius: float) -> str:
    return f"{celsius:.0f}°C ({_c_to_f(celsius)}°F)"


def assess(
    forecast: WeatherForecast,
    *,
    hot_c: float = DEFAULT_HOT_C,
    cold_c: float = DEFAULT_COLD_C,
) -> WeatherAssessment:
    """Bird-safety read of a forecast: warnings + concrete care advice."""
    is_hot = forecast.high_c >= hot_c
    is_cold = forecast.low_c <= cold_c
    is_wet = forecast.precip_prob is not None and forecast.precip_prob >= 70

    warnings: list[str] = []
    advice: list[str] = []

    if is_hot:
        warnings.append(
            f"🔥 Hot day ahead — high of {_temp(forecast.high_c)}. Watch the flock for "
            "heat stress (panting, wings held away from the body)."
        )
        advice.append(
            "Keep cages out of direct sun and in the coolest, airy spot; offer extra "
            "fresh water and a shallow bath or light misting; never leave a bird in a "
            "parked car or a closed sunny room."
        )
    if is_cold:
        warnings.append(
            f"❄️ Cold night ahead — low of {_temp(forecast.low_c)}. That's below the "
            "comfortable range for tropical cage birds."
        )
        advice.append(
            "Move cages to the warmest, draft-free room overnight, cover the cage, and "
            "consider a bird-safe heat source. Below ~4 °C is dangerous for them."
        )
    if is_wet:
        advice.append(
            f"High chance of rain ({forecast.precip_prob}%) — keep any outdoor or "
            "balcony cages dry and sheltered."
        )
    if not (is_hot or is_cold):
        advice.append(
            f"Comfortable range today ({_temp(forecast.low_c)}–{_temp(forecast.high_c)}) "
            "— good conditions for the flock."
        )

    return WeatherAssessment(
        is_hot=is_hot,
        is_cold=is_cold,
        is_wet=is_wet,
        warnings=warnings,
        advice=advice,
    )


def summarize(
    forecast: WeatherForecast,
    *,
    hot_c: float = DEFAULT_HOT_C,
    cold_c: float = DEFAULT_COLD_C,
) -> str:
    """The /weather reply: outlook line, warnings, and bird advice."""
    assessment = assess(forecast, hot_c=hot_c, cold_c=cold_c)
    when = f" for {forecast.date_label}" if forecast.date_label else ""
    lines = [f"🌤️ **Weather{when}**"]
    if forecast.current_c is not None:
        lines.append(f"Now: {_temp(forecast.current_c)}, {forecast.condition}.")
    lines.append(
        f"Today: {forecast.condition}, high {_temp(forecast.high_c)}, "
        f"low {_temp(forecast.low_c)} overnight."
    )
    if forecast.precip_prob is not None:
        lines.append(f"Rain chance: {forecast.precip_prob}%.")
    for warning in assessment.warnings:
        lines.append(warning)
    if assessment.advice:
        lines.append("")
        lines.append("**For the birds:** " + " ".join(assessment.advice))
    return "\n".join(lines)


class WeatherMonitor:
    """Polls the forecast several times a day and warns on hot days / cold nights.

    Each distinct warning (hot day, cold night) is sent at most once per local day
    so repeated polls don't nag. ``fetch`` is injected (already bound to the
    location) so this is testable without the network.
    """

    def __init__(
        self,
        notify_all: Callable[[str], None],
        stop_event: threading.Event,
        fetch: Callable[[], WeatherForecast | None],
        *,
        hot_c: float = DEFAULT_HOT_C,
        cold_c: float = DEFAULT_COLD_C,
        now: Callable[[], datetime] = now_ph,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._notify_all = notify_all
        self._stop = stop_event
        self._fetch = fetch
        self._hot_c = hot_c
        self._cold_c = cold_c
        self._now = now
        self._poll = poll_seconds
        self._fired: dict[str, str] = {}  # "hot"/"cold" -> day key it last warned for

    def start(self) -> None:
        threading.Thread(target=self._run, name="weather-monitor", daemon=True).start()

    def _run(self) -> None:
        LOGGER.info("Weather monitor started (poll %.0fs)", self._poll)
        # Check once shortly after boot, then on the poll cadence.
        self._tick()
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                break
            try:
                self._tick()
            except Exception:
                LOGGER.exception("Weather monitor tick failed")

    def _tick(self) -> None:
        forecast = self._fetch()
        if forecast is None:
            return
        assessment = assess(forecast, hot_c=self._hot_c, cold_c=self._cold_c)
        day_key = self._now().date().isoformat()
        if assessment.is_hot and self._fired.get("hot") != day_key:
            self._fired["hot"] = day_key
            self._warn(assessment, forecast, "hot")
        if assessment.is_cold and self._fired.get("cold") != day_key:
            self._fired["cold"] = day_key
            self._warn(assessment, forecast, "cold")

    def _warn(self, assessment: WeatherAssessment, forecast: WeatherForecast, kind: str) -> None:
        warning = next(
            (w for w in assessment.warnings if (kind == "hot") == w.startswith("🔥")),
            None,
        )
        if warning is None:
            return
        advice = " ".join(assessment.advice)
        message = warning + (f"\n\n**For the birds:** {advice}" if advice else "")
        try:
            self._notify_all(message)
        except Exception:
            LOGGER.exception("Weather warning failed to send")
        LOGGER.info("Weather warning sent: %s", kind)
