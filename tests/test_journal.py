from __future__ import annotations

from datetime import date, datetime

from lib.journal import MemoryEntry, append_entry, load_entries, load_recent, memory_path


def test_append_creates_dated_file_with_header(tmp_path) -> None:
    entry = MemoryEntry(datetime(2026, 6, 25, 14, 32), ["percy", "matcha"], "Percy preens by Matcha.", ["a.jpg", "b.jpg"])
    path = append_entry(tmp_path, entry)
    assert path == memory_path(tmp_path, date(2026, 6, 25))
    text = path.read_text()
    assert text.startswith("# Aviary memories — 2026-06-25")
    assert "## 14:32 | percy, matcha" in text
    assert "> photo: a.jpg" in text and "> photo: b.jpg" in text


def test_append_then_load_roundtrip_multiple_photos(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 5), ["bambi"], "Bambi eats.", ["b1.jpg", "b2.jpg"]))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 40), ["percy"], "Percy naps."))
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert [e.time.strftime("%H:%M") for e in entries] == ["09:05", "09:40"]
    assert entries[0].photos == ["b1.jpg", "b2.jpg"]
    assert entries[1].note == "Percy naps."
    assert entries[1].photos == []


def test_quiet_entry_has_no_birds(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 3, 0), [], "All quiet."))
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert entries[0].birds == []


def test_load_recent_filters_by_time_and_bird(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["percy"], "early percy"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 12, 0), ["matcha"], "noon matcha"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 13, 0), ["percy"], "afternoon percy"))

    window = load_recent(tmp_path, datetime(2026, 6, 25, 11, 0), datetime(2026, 6, 25, 14, 0))
    assert [e.note for e in window] == ["noon matcha", "afternoon percy"]

    percy = load_recent(
        tmp_path, datetime(2026, 6, 25, 0, 0), datetime(2026, 6, 25, 23, 59), birds={"percy"}
    )
    assert [e.note for e in percy] == ["early percy", "afternoon percy"]


def test_load_missing_day_is_empty(tmp_path) -> None:
    assert load_entries(tmp_path, date(2026, 1, 1)) == []


def test_load_recent_spans_multiple_days(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 20, 9, 0), ["jynx"], "monday"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 23, 9, 0), ["jynx"], "thursday"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 0), ["jynx"], "saturday"))
    week = load_recent(tmp_path, datetime(2026, 6, 19, 0, 0), datetime(2026, 6, 25, 23, 59))
    assert [e.note for e in week] == ["monday", "thursday", "saturday"]
