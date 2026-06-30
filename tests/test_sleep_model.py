from __future__ import annotations

from datetime import date, datetime

from lib.sleep.model import (
    Disturbance,
    SleepNight,
    append_night,
    clear_current,
    from_json,
    load_current,
    load_nights,
    load_recent,
    save_current,
    to_json,
)


def _night(day: date, *, out_h: int = 20, score: int | None = 90, finalized: bool = True) -> SleepNight:
    return SleepNight(
        night_of=day,
        lights_out=datetime(day.year, day.month, day.day, out_h, 40),
        first_light=datetime(day.year, day.month, day.day + 1 if day.day < 28 else day.day, 7, 30),
        dark_minutes=11 * 60,
        disturbances=[Disturbance(datetime(day.year, day.month, day.day, 23, 10), "light", "brief")],
        max_motion=8.0,
        camera_count_at_dark=2,
        camera_count_at_wake=2,
        score=score,
        components={"duration": 1.0, "consistency": 1.0, "darkness": 0.85, "disturbance": 1.0},
        confidence=0.9,
        notes=["on schedule"],
        summary="A good night.",
        finalized=finalized,
    )


def test_to_from_json_roundtrips_all_fields() -> None:
    night = _night(date(2026, 6, 25))
    restored = from_json(to_json(night))
    assert restored == night
    # None fields survive too.
    blank = SleepNight(night_of=date(2026, 6, 25))
    assert from_json(to_json(blank)) == blank


def test_append_files_by_night_of_month(tmp_path) -> None:
    append_night(tmp_path, _night(date(2026, 6, 25)))
    append_night(tmp_path, _night(date(2026, 7, 1)))
    assert (tmp_path / "2026-06.jsonl").exists()
    assert (tmp_path / "2026-07.jsonl").exists()
    # One line per night in its month file.
    assert len((tmp_path / "2026-06.jsonl").read_text().splitlines()) == 1


def test_load_recent_newest_first_across_months(tmp_path) -> None:
    for day in (date(2026, 6, 24), date(2026, 6, 25), date(2026, 7, 1)):
        append_night(tmp_path, _night(day))
    recent = load_recent(tmp_path, 2)
    assert [n.night_of for n in recent] == [date(2026, 7, 1), date(2026, 6, 25)]


def test_load_recent_skips_unfinalized(tmp_path) -> None:
    append_night(tmp_path, _night(date(2026, 6, 24), finalized=False))
    append_night(tmp_path, _night(date(2026, 6, 25)))
    assert [n.night_of for n in load_recent(tmp_path, 5)] == [date(2026, 6, 25)]


def test_load_nights_filters_by_date_window(tmp_path) -> None:
    for day in (date(2026, 6, 23), date(2026, 6, 25), date(2026, 6, 27)):
        append_night(tmp_path, _night(day))
    got = load_nights(tmp_path, date(2026, 6, 24), date(2026, 6, 26))
    assert [n.night_of for n in got] == [date(2026, 6, 25)]


def test_current_sidecar_roundtrip_and_clear(tmp_path) -> None:
    night = SleepNight(night_of=date(2026, 6, 25), lights_out=datetime(2026, 6, 25, 20, 30))
    assert load_current(tmp_path) is None  # nothing yet
    save_current(tmp_path, night)
    assert load_current(tmp_path) == night
    clear_current(tmp_path)
    assert load_current(tmp_path) is None


def test_corrupt_sidecar_returns_none(tmp_path) -> None:
    (tmp_path / "_current.json").write_text("{not valid json", encoding="utf-8")
    assert load_current(tmp_path) is None  # never raises
