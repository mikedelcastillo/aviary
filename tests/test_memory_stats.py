from __future__ import annotations

import json
from datetime import datetime

from lib.journal import MemoryDetection, MemoryEntry, MemoryObservation, append_entry
from lib.memory_stats import format_memory_stats, gather_memory_stats


def _obs(photo: str, *, vlm: str = "qwen2.5vl:7b", activity: str = "eating") -> MemoryObservation:
    return MemoryObservation(
        camera="Cage",
        birds=["percy"],
        note="on the perch",
        photo=photo,
        detections=[MemoryDetection(label="percy", confidence=0.9, bbox=(1, 2, 3, 4), activity=activity)],
        detector_model="live-019.pt",
        vlm_model=vlm,
    )


def _store(tmp_path):
    memories = tmp_path / "memories"
    images = memories / "images"
    images.mkdir(parents=True)
    photo_ok = images / "a.jpg"
    photo_ok.write_bytes(b"x" * 1000)
    photo_gone = images / "gone.jpg"  # referenced but never written
    # Day 1: one decorated observation + one awaiting backfill (no vlm_model).
    append_entry(memories, MemoryEntry(
        datetime(2026, 7, 3, 9, 0), ["percy"], "note",
        photos=[str(photo_ok)],
        observations=[_obs(str(photo_ok))],
    ))
    append_entry(memories, MemoryEntry(
        datetime(2026, 7, 3, 10, 0), ["percy"], "note",
        photos=[str(photo_ok)],
        observations=[_obs(str(photo_ok), vlm="", activity="")],
    ))
    # Day 2: a pre-v3 record (hand-written raw line) with a missing photo.
    day2 = memories / "2026-07-04.jsonl"
    with day2.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "version": 1,
            "time": "2026-07-04T08:00:00",
            "birds": ["bambi"],
            "note": "seen",
            "photos": [str(photo_gone)],
            "observations": [{"camera": "", "birds": ["bambi"], "note": "seen", "photo": str(photo_gone)}],
        }) + "\n")
    return memories


NOW = lambda: datetime(2026, 7, 4, 12, 0)  # noqa: E731 — fixed clock for the backfill window


def test_gather_counts_days_entries_and_debt(tmp_path) -> None:
    stats = gather_memory_stats(_store(tmp_path), pending_count=2, now=NOW)
    assert stats.days == 2
    assert (stats.first_day, stats.last_day) == ("2026-07-03", "2026-07-04")
    assert stats.entries == 3
    assert stats.observations == 3
    assert stats.bird_records == 2  # the pre-v3 obs has no detections
    assert stats.activity_tagged == 1  # only the decorated one
    assert stats.undecorated == 1  # the vlm-less v3 obs awaits backfill
    assert stats.pre_v3_entries == 1  # migrate's job, not the backfill's
    assert stats.photos_referenced == 3
    assert stats.photos_missing == 1
    assert stats.image_count == 1
    assert stats.image_bytes == 1000
    assert stats.memories_bytes > 1000  # images + day files
    assert stats.pending_messages == 2


def test_gather_includes_data_dir_total(tmp_path) -> None:
    memories = _store(tmp_path)
    (tmp_path / "detection").mkdir()
    (tmp_path / "detection" / "log.json").write_bytes(b"y" * 500)
    stats = gather_memory_stats(memories, data_dir=tmp_path, now=NOW)
    assert stats.data_bytes >= stats.memories_bytes + 500


def test_format_shows_debt_and_missing_photos(tmp_path) -> None:
    text = format_memory_stats(gather_memory_stats(_store(tmp_path), pending_count=1, now=NOW))
    assert text.startswith("🧠 Memory")
    assert "Days: 2 (2026-07-03 → 2026-07-04)" in text
    assert "1 awaiting VLM backfill" in text
    assert "1 pre-v3 (run memory-migrate)" in text
    assert "1 queued message(s)" in text
    assert "1 images" in text
    assert "1 of 3 referenced photo file(s) missing" in text


def test_format_all_caught_up(tmp_path) -> None:
    memories = tmp_path / "memories"
    memories.mkdir()
    photo = memories / "p.jpg"
    photo.write_bytes(b"z")
    append_entry(memories, MemoryEntry(
        datetime(2026, 7, 4, 9, 0), ["percy"], "note",
        photos=[str(photo)], observations=[_obs(str(photo))],
    ))
    text = format_memory_stats(gather_memory_stats(memories, now=NOW))
    assert "all caught up ✅" in text
    assert "missing" not in text


def test_gather_empty_store(tmp_path) -> None:
    stats = gather_memory_stats(tmp_path / "nowhere")
    assert stats.days == 0 and stats.entries == 0
    assert "Days: 0" in format_memory_stats(stats)


def test_old_undecorated_is_auto_backfill_work_with_unbounded_scan(tmp_path) -> None:
    # The backfill worker scans the whole history by default
    # (MEMORY_BACKFILL_DAYS=None), so even a month-old undecorated observation
    # is "awaiting (auto)" — nothing is left to a manual memory-migrate run.
    memories = tmp_path / "memories"
    memories.mkdir()
    photo = memories / "old.jpg"
    photo.write_bytes(b"x")
    append_entry(memories, MemoryEntry(
        datetime(2026, 6, 25, 9, 0), ["percy"], "note",
        photos=[str(photo)],
        observations=[_obs(str(photo), vlm="", activity="")],
    ))
    stats = gather_memory_stats(memories, now=NOW)
    assert stats.undecorated == 1
    assert stats.undecorated_stale == 0


def test_bounded_backfill_window_still_splits_stale_work(tmp_path, monkeypatch) -> None:
    # With a bounded window the old split survives: older-than-window
    # undecorated work is labeled for memory-migrate, not "(auto)".
    monkeypatch.setattr("lib.memory_stats.MEMORY_BACKFILL_DAYS", 3)
    memories = tmp_path / "memories"
    memories.mkdir()
    photo = memories / "old.jpg"
    photo.write_bytes(b"x")
    append_entry(memories, MemoryEntry(
        datetime(2026, 6, 25, 9, 0), ["percy"], "note",
        photos=[str(photo)],
        observations=[_obs(str(photo), vlm="", activity="")],
    ))
    stats = gather_memory_stats(memories, now=NOW)
    assert stats.undecorated == 0
    assert stats.undecorated_stale == 1
    text = format_memory_stats(stats)
    assert "1 older undecorated (run memory-migrate)" in text
    assert "awaiting VLM backfill" not in text
