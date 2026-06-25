from __future__ import annotations

import threading
from datetime import datetime

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
    def chat(self, model, messages, **kwargs):
        return "Percy preened on the perch."


def row(label, camera="camera-192.168.1.8", since=1.0):
    return {"camera": camera, "label": label, "since": since}


def _maker(memories, registry, notifier, now_dt, clock_val, *, describe=lambda img: "perched"):
    return MemoryMaker(
        memories,
        registry,
        grab_frame=lambda cam: b"\xff\xd8jpeg-" + cam.encode(),
        describe_frame=describe,
        client=FakeClient(),
        llm_model="qwen3:4b",
        notifier=notifier,
        stop_event=threading.Event(),
        interval_seconds=300,
        poll_seconds=30,
        fresh_seconds=15,
        camera_display=lambda n: "Big Cage",
        clock=lambda: clock_val,
        now=lambda: now_dt,
    )


def test_report_saves_images_writes_memory_and_broadcasts(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)

    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True

    # Photo(s) sent, image(s) saved to the memory image store, memory written.
    assert notifier.photos and notifier.tracked
    images = list((memories / "images").glob("*.jpg"))
    assert images, "expected a memory image to be saved"
    entries = load_entries(memories, now_dt.date())
    assert entries and entries[0].birds == ["percy"]
    assert entries[0].photos and entries[0].photos[0].endswith(".jpg")


def test_tick_reports_on_new_bird(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._tick()
    assert notifier.tracked


def test_tick_edits_in_place_when_quiet(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 3, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([]), notifier, now_dt, 1000.0)
    maker._last_report_at = 1000.0 - 400  # the 5-min beat is due
    maker._activity_msgs = {"A": 100}
    maker._tick()
    assert notifier.tracked == []
    assert notifier.edits and "quiet" in notifier.edits[0][2].lower()
