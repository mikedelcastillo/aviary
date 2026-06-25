from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

from lib.journal import load_entries
from lib.memory_maker import MemoryMaker


class FakeNotifier:
    user_ids = ["A"]

    def __init__(self) -> None:
        self.tracked: list[str] = []
        self.edits: list[tuple] = []
        self.photos: list = []

    def send_photo(self, uid, image, caption):
        self.photos.append(caption)
        return True

    def broadcast_text_tracked(self, text):
        self.tracked.append(text)
        return {"A": 100}

    def edit_message_text(self, chat_id, message_id, text):
        self.edits.append((chat_id, message_id, text))
        return True


class FakeRegistry:
    def __init__(self, rows):
        self._rows = rows

    def snapshot(self):
        return list(self._rows)


class FakeClient:
    def generate(self, model, prompt, *, images=None, timeout_seconds=None, **kwargs):
        return "doing bird things"

    def chat(self, model, messages, **kwargs):
        return "Percy preened with Matcha."


def _write_sighting(collect_dir, label, conf, when: datetime) -> None:
    folder = collect_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{int(when.timestamp())}"
    (folder / f"{stem}.jpg").write_bytes(b"\xff\xd8jpeg")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "object": label,
                "camera": {"name": "camera-192.168.1.8"},
                "collected_at": when.replace(tzinfo=timezone.utc).isoformat(),
                "frame": {"width": 100, "height": 100},
                "detection": {"confidence": conf, "bbox_xyxy": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}},
            }
        )
    )


def row(label, since=1.0):
    return {"camera": "camera-192.168.1.8", "label": label, "since": since}


def _maker(collect, memories, registry, notifier, now_dt, clock_val):
    return MemoryMaker(
        collect, memories, registry, FakeClient(), "qwen3:4b", "qwen2.5vl:7b", notifier,
        threading.Event(), interval_seconds=300, poll_seconds=30, fresh_seconds=15,
        camera_display=lambda n: "Big Cage", clock=lambda: clock_val, now=lambda: now_dt,
    )


def test_report_writes_memory_and_broadcasts(tmp_path) -> None:
    collect = tmp_path / "collect"
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    _write_sighting(collect, "percy", 0.9, now_dt - timedelta(minutes=2))
    notifier = FakeNotifier()
    maker = _maker(collect, memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._window_start = now_dt - timedelta(minutes=5)

    maker._report(frozenset({"percy"}))

    assert notifier.tracked and "Percy" in notifier.tracked[0]
    assert notifier.photos  # photo(s) sent individually
    entries = load_entries(memories, now_dt.date())
    assert entries and "percy" in entries[0].birds


def test_tick_reports_on_new_bird(tmp_path) -> None:
    collect = tmp_path / "collect"
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    _write_sighting(collect, "percy", 0.9, now_dt - timedelta(minutes=1))
    notifier = FakeNotifier()
    maker = _maker(collect, memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._window_start = now_dt - timedelta(minutes=5)

    maker._tick()  # percy is new -> immediate report

    assert notifier.tracked


def test_tick_edits_in_place_when_quiet(tmp_path) -> None:
    collect = tmp_path / "collect"
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 3, 0)
    notifier = FakeNotifier()
    maker = _maker(collect, memories, FakeRegistry([]), notifier, now_dt, 1000.0)
    maker._last_report_at = 1000.0 - 400  # the 5-min beat is due
    maker._activity_msgs = {"A": 100}

    maker._tick()  # quiet + due -> edit the last message, no new broadcast

    assert notifier.tracked == []
    assert notifier.edits and "quiet" in notifier.edits[0][2].lower()
