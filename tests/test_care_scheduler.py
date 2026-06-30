from __future__ import annotations

import threading
from datetime import datetime, time

from lib.care_scheduler import CareScheduler

# Fixed sun times so tests don't depend on the real date: sunrise 05:30, sunset 18:30.
SUN = lambda day: (time(5, 30), time(18, 30))
THURSDAY = datetime(2026, 6, 25)  # weekday() == 3
SUNDAY = datetime(2026, 6, 28)    # weekday() == 6


class _Light:
    def __init__(self, dark: bool = False) -> None:
        self.dark = dark

    def all_ir(self) -> bool:
        return self.dark


def _make(sun=SUN, **kwargs):
    sent: list[str] = []
    light = _Light()
    scheduler = CareScheduler(
        sent.append,
        threading.Event(),
        all_ir=light.all_ir,
        camera_count=lambda: 2,
        sun_times=sun,
        **kwargs,
    )
    return scheduler, sent, light


def test_morning_fires_at_sunrise_anchor_once_per_day() -> None:
    scheduler, sent, _ = _make()
    scheduler._tick(THURSDAY.replace(hour=5, minute=45))
    assert sum("Good morning" in m for m in sent) == 1
    # A second tick in the same window does not re-fire.
    scheduler._tick(THURSDAY.replace(hour=5, minute=55))
    assert sum("Good morning" in m for m in sent) == 1


def test_morning_fires_on_observed_lights_on() -> None:
    scheduler, sent, light = _make()
    light.dark = True
    scheduler._tick(THURSDAY.replace(hour=5, minute=5))  # still dark, before sunrise window
    assert sent == []
    light.dark = False  # the room lights up
    scheduler._tick(THURSDAY.replace(hour=5, minute=10))
    assert any("Good morning" in m for m in sent)


def test_food_pickup_fires_two_hours_after_morning() -> None:
    scheduler, sent, _ = _make()
    scheduler._tick(THURSDAY.replace(hour=5, minute=45))  # morning
    scheduler._tick(THURSDAY.replace(hour=6, minute=30))  # +45m: too soon
    assert not any("pull any leftover" in m for m in sent)
    scheduler._tick(THURSDAY.replace(hour=7, minute=50))  # +2h05m
    assert any("pull any leftover" in m for m in sent)


def test_bedtime_fires_on_observed_lights_off_in_evening() -> None:
    scheduler, sent, light = _make()
    light.dark = False
    scheduler._tick(THURSDAY.replace(hour=18, minute=0))  # establish "light"
    light.dark = True
    scheduler._tick(THURSDAY.replace(hour=18, minute=40))
    assert any("Lights out" in m for m in sent)


def test_bedtime_clock_fallback_without_sun_times() -> None:
    scheduler, sent, _ = _make(sun=lambda day: (None, None))  # no sun data -> 20:00 fallback
    scheduler._tick(THURSDAY.replace(hour=20, minute=15))
    assert any("Lights out" in m for m in sent)


def test_no_backfire_when_started_late_evening() -> None:
    # Booting at 22:00 must not dump morning/midday/wind-down all at once.
    scheduler, sent, _ = _make()
    scheduler._tick(THURSDAY.replace(hour=22, minute=0))
    assert sent == []


def test_winddown_fires_before_sunset_not_after() -> None:
    # sunset 18:30 -> wind-down anchor 17:30. The "~1h before sunset" nudge must
    # land before sunset, never after it.
    scheduler, sent, _ = _make()
    scheduler._tick(THURSDAY.replace(hour=17, minute=40))  # inside the pre-sunset window
    assert any("Winding down" in m for m in sent)

    later, sent_late, _ = _make()
    later._tick(THURSDAY.replace(hour=18, minute=45))  # just after sunset
    assert not any("Winding down" in m for m in sent_late)


def test_morning_folds_in_last_night_sleep() -> None:
    scheduler, sent, _ = _make(sleep_summary=lambda: "😴 They slept 11h of dark, score 90.")
    scheduler._tick(THURSDAY.replace(hour=5, minute=45))  # morning fires
    morning = next(m for m in sent if "Good morning" in m)
    assert "They slept 11h" in morning


def test_weekly_clean_folds_in_care_tip() -> None:
    scheduler, sent, _ = _make(weekly_tip=lambda: "💡 Care tip: wash dishes daily.")
    scheduler._tick(SUNDAY.replace(hour=13, minute=10))
    clean = next(m for m in sent if "deep-clean" in m)
    assert "Care tip" in clean


def test_weekly_clean_fires_on_weekly_day_only() -> None:
    scheduler, sent, _ = _make()
    scheduler._tick(THURSDAY.replace(hour=13, minute=10))  # not the weekly day
    assert not any("deep-clean" in m for m in sent)
    scheduler2, sent2, _ = _make()
    scheduler2._tick(SUNDAY.replace(hour=13, minute=10))
    assert any("deep-clean" in m for m in sent2)
    # Once per week.
    scheduler2._tick(SUNDAY.replace(hour=13, minute=40))
    assert sum("deep-clean" in m for m in sent2) == 1
