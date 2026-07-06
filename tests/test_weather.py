from __future__ import annotations

import threading
from datetime import datetime

from lib.weather import (
    WeatherForecast,
    WeatherMonitor,
    assess,
    has_location,
    parse_forecast,
    summarize,
)


def _payload(high, low, *, code=2, precip=20, current=None, humidity=None):
    return {
        "current": (
            {"temperature_2m": current, "relative_humidity_2m": humidity, "weather_code": code}
            if current is not None
            else {}
        ),
        "daily": {
            "time": ["2026-06-30"],
            "temperature_2m_max": [high],
            "temperature_2m_min": [low],
            "precipitation_probability_max": [precip],
            "weather_code": [code],
        },
    }


def test_parse_forecast_extracts_fields():
    fc = parse_forecast(_payload(33.0, 25.0, code=80, precip=60, current=30.5, humidity=72))
    assert fc is not None
    assert fc.high_c == 33.0
    assert fc.low_c == 25.0
    assert fc.date_label == "2026-06-30"
    assert fc.current_c == 30.5
    assert fc.humidity == 72
    assert fc.precip_prob == 60
    assert "shower" in fc.condition  # WMO 80 -> rain showers


def test_parse_forecast_missing_daily_is_none():
    assert parse_forecast({"current": {}}) is None
    assert parse_forecast({"daily": {"temperature_2m_max": []}}) is None


def test_assess_hot_day():
    a = assess(WeatherForecast("2026-06-30", high_c=34.0, low_c=26.0, condition="clear"), hot_c=32.0, cold_c=10.0)
    assert a.is_hot and not a.is_cold
    assert any(w.startswith("🔥") for w in a.warnings)
    assert a.advice  # heat advice present


def test_assess_cold_night():
    a = assess(WeatherForecast("2026-06-30", high_c=16.0, low_c=8.0, condition="clear"), hot_c=32.0, cold_c=10.0)
    assert a.is_cold and not a.is_hot
    assert any(w.startswith("❄️") for w in a.warnings)


def test_assess_comfortable_has_no_warnings():
    a = assess(WeatherForecast("2026-06-30", high_c=27.0, low_c=20.0, condition="partly cloudy"), hot_c=32.0, cold_c=10.0)
    assert not a.is_hot and not a.is_cold
    assert a.warnings == []
    assert any("Comfortable" in line for line in a.advice)


def test_summarize_includes_outlook_and_advice():
    fc = WeatherForecast("2026-06-30", high_c=34.0, low_c=26.0, condition="clear sky", current_c=31.0, precip_prob=10)
    text = summarize(fc, hot_c=32.0, cold_c=10.0)
    assert "high" in text and "overnight" in text
    assert "For the birds:" in text
    assert "🔥" in text  # hot warning folded in


def test_summarize_includes_sun_times_when_given():
    from datetime import time
    fc = WeatherForecast("2026-06-30", high_c=28.0, low_c=22.0, condition="clear", current_c=25.0)
    text = summarize(fc, hot_c=32.0, cold_c=10.0, sunrise=time(5, 47), sunset=time(18, 21))
    # "what time was sunrise?" (routes to weather) gets a real answer.
    assert "Sunrise 5:47 AM" in text
    assert "Sunset 6:21 PM" in text


def test_summarize_omits_sun_line_without_times():
    fc = WeatherForecast("2026-06-30", high_c=28.0, low_c=22.0, condition="clear")
    text = summarize(fc, hot_c=32.0, cold_c=10.0)
    assert "Sunrise" not in text and "Sunset" not in text


def test_has_location():
    assert has_location(14.6, 121.0) is True
    assert has_location(0.0, 0.0) is False


def test_monitor_warns_once_per_day_then_again_next_day():
    sent: list[str] = []
    today = {"d": datetime(2026, 6, 30, 7, 0)}
    hot = WeatherForecast("2026-06-30", high_c=35.0, low_c=27.0, condition="clear")

    monitor = WeatherMonitor(
        sent.append,
        threading.Event(),
        lambda: hot,
        hot_c=32.0,
        cold_c=10.0,
        now=lambda: today["d"],
    )

    monitor._tick()
    monitor._tick()  # same day -> deduped
    assert len(sent) == 1
    assert sent[0].startswith("🔥")

    today["d"] = datetime(2026, 7, 1, 7, 0)  # new day -> warns again
    monitor._tick()
    assert len(sent) == 2


def test_monitor_silent_when_fetch_fails_or_comfortable():
    sent: list[str] = []
    comfortable = WeatherForecast("2026-06-30", high_c=26.0, low_c=19.0, condition="clear")
    WeatherMonitor(sent.append, threading.Event(), lambda: None, now=lambda: datetime(2026, 6, 30))._tick()
    WeatherMonitor(sent.append, threading.Event(), lambda: comfortable, now=lambda: datetime(2026, 6, 30))._tick()
    assert sent == []
