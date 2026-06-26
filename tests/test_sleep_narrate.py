from __future__ import annotations

from datetime import date, datetime

from lib.sleep.model import Disturbance, SleepNight
from lib.sleep.narrate import (
    format_last,
    format_morning,
    format_status_line,
    format_week,
    llm_summary,
    sleep_streak,
)


def _night(day=date(2026, 6, 25), score=91, dark_min=665, frights=0, lights=0, consistency=1.0) -> SleepNight:
    dist = [Disturbance(datetime(2026, 6, day.day, 23, 10), "light", minutes=5) for _ in range(lights)]
    dist += [Disturbance(datetime(2026, 6, day.day + 1, 2, 14), "night_fright", "possible") for _ in range(frights)]
    return SleepNight(
        night_of=day,
        lights_out=datetime(day.year, day.month, day.day, 20, 38),
        first_light=datetime(day.year, day.month, day.day + 1, 7, 43),
        dark_minutes=dark_min,
        disturbances=dist,
        score=score,
        components={"duration": 1.0, "consistency": consistency, "darkness": 0.9, "disturbance": 1.0},
        confidence=0.9,
        finalized=True,
    )


def test_format_last_has_score_dark_and_schedule() -> None:
    text = format_last(_night(score=91, lights=1))
    assert "91/100" in text
    assert "11h05m" in text  # 665 min
    assert "sweet spot" in text  # 11.08h is in 10-12
    assert "8:38pm" in text and "7:43am" in text
    assert "On schedule" in text


def test_format_last_flags_night_fright() -> None:
    text = format_last(_night(frights=1))
    assert "night-fright" in text
    assert "pulled feathers" in text


def test_format_last_short_night_phrasing() -> None:
    text = format_last(_night(dark_min=7 * 60))  # 7h
    assert "short" in text.lower()


def test_sleep_streak_counts_leading_good_nights() -> None:
    # Newest-first: a run of good nights breaks at the first sub-80 night.
    nights = [_night(score=s) for s in (91, 84, 80, 70, 95)]
    assert sleep_streak(nights) == 3
    assert sleep_streak([_night(score=50)]) == 0
    assert sleep_streak([]) == 0


def test_format_last_shows_streak() -> None:
    assert "good nights in a row" in format_last(_night(score=91), streak=3)
    assert "good nights in a row" not in format_last(_night(score=91), streak=1)


def test_format_morning_one_liner() -> None:
    text = format_morning(_night(score=89))
    assert text.startswith("😴 ")
    assert "Sleep report" in text
    assert "89/100" in text


def test_format_week_trend_and_extremes() -> None:
    nights = [_night(date(2026, 6, d), score=s) for d, s in [(22, 96), (23, 80), (24, 68), (25, 90)]]
    text = format_week(nights)
    assert "Avg score" in text
    assert "96" in text and "68" in text  # best + worst
    assert "Dark hours: avg" in text


def test_format_week_empty() -> None:
    assert "No finished nights" in format_week([])


def test_format_last_cold_start_has_no_usual_to_compare() -> None:
    night = _night()
    night.baselined = False  # first night, no rolling baseline yet
    text = format_last(night)
    assert "Still learning their routine" in text
    assert "off their usual" not in text


def test_status_line_clamps_negative_time() -> None:
    night = SleepNight(night_of=date(2026, 6, 25), lights_out=datetime(2026, 6, 25, 20, 0))
    line = format_status_line(night, datetime(2026, 6, 25, 19, 0))  # now BEFORE lights_out
    assert line is not None and "0h00m" in line  # clamped, never negative


def test_status_line_while_night_open() -> None:
    night = SleepNight(night_of=date(2026, 6, 25), lights_out=datetime(2026, 6, 25, 20, 0))
    line = format_status_line(night, datetime(2026, 6, 25, 23, 12))
    assert line is not None and "3h12m" in line
    assert format_status_line(None, datetime(2026, 6, 25, 23, 0)) is None


def test_llm_summary_falls_back_on_error() -> None:
    class BoomClient:
        def chat(self, *a, **k):
            raise RuntimeError("ollama down")

    out = llm_summary(BoomClient(), "m", _night(score=89))
    assert out == format_morning(_night(score=89))  # deterministic fallback

    # No client -> deterministic too.
    assert llm_summary(None, "m", _night(score=89)) == format_morning(_night(score=89))
