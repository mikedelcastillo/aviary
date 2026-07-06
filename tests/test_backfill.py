from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np

from lib.ai.client import OllamaUnavailableError
from lib.backfill import (
    GAVE_UP_NOTICE,
    MESSAGE_MAX_ATTEMPTS,
    PHOTO_MAX_ATTEMPTS,
    LlmBackfillWorker,
    build_memory_backfill,
)
from lib.journal import (
    MemoryEntry,
    append_entry,
    load_day_records,
    load_entries,
    memory_jsonl_path,
)
from lib.memory_build import build_observation
from lib.pending import PendingMessageStore


# ---------------------------------------------------------------------------
# PART A — LlmBackfillWorker drain logic (no thread; _drain_messages directly)
# ---------------------------------------------------------------------------


class FakeHealth:
    """Stand-in for the Ollama health monitor: an `available` flag + on_change."""

    def __init__(self, available: bool = True, served: set[str] | None = None) -> None:
        self.available = available
        self.served = served  # None = optimistic (serves everything)
        self.listeners: list = []

    def on_change(self, listener) -> None:
        self.listeners.append(listener)

    def has_model(self, name: str) -> bool:
        return True if self.served is None else name in self.served


class RecordingReplay:
    """Replay callable that records calls and can fail per-text."""

    def __init__(self, fail_texts: set[str] | None = None, fail_all: bool = False) -> None:
        self.calls: list[tuple[int, str, object]] = []
        self.fail_texts = fail_texts or set()
        self.fail_all = fail_all

    def __call__(self, chat_id: int, text: str, created) -> None:
        self.calls.append((chat_id, text, created))
        if self.fail_all or text in self.fail_texts:
            raise RuntimeError("replay boom")


def _worker(store, health, replay, notify=None) -> LlmBackfillWorker:
    return LlmBackfillWorker(
        store, health, threading.Event(), replay=replay, notify=notify
    )


def _queue(store: PendingMessageStore, texts: list[str], chat_id: int = 7) -> None:
    when = datetime(2026, 7, 1, 10, 0)
    for i, text in enumerate(texts):
        store.add(chat_id, text, when + timedelta(minutes=i))


def test_drain_replays_oldest_first_and_removes_each(tmp_path) -> None:
    # Happy path: every queued message is replayed in chronological order and
    # its file is deleted on success — nothing typed during an outage is lost.
    store = PendingMessageStore(tmp_path / "pending")
    _queue(store, ["first", "second", "third"])
    replay = RecordingReplay()

    _worker(store, FakeHealth(available=True), replay)._drain_messages()

    assert [text for _, text, _ in replay.calls] == ["first", "second", "third"]
    assert store.count() == 0


def test_replay_failure_bumps_attempts_keeps_item_and_stops_cycle(tmp_path) -> None:
    # A failing replay likely means the cluster dropped again: the item must be
    # retained (attempts+1) and the drain must stop for this cycle — the second
    # message is NOT attempted.
    store = PendingMessageStore(tmp_path / "pending")
    _queue(store, ["broken", "later"])
    replay = RecordingReplay(fail_all=True)

    _worker(store, FakeHealth(available=True), replay)._drain_messages()

    assert [text for _, text, _ in replay.calls] == ["broken"]  # drain stopped
    items = store.items()
    assert [i.text for i in items] == ["broken", "later"]  # both retained
    assert items[0].attempts == 1
    assert items[1].attempts == 0


def test_gives_up_after_max_attempts_with_notice_and_continues(tmp_path) -> None:
    # After MESSAGE_MAX_ATTEMPTS failures the user is told (never silence), the
    # item is removed, and the drain moves on so one poison message can't block
    # the rest of the queue.
    store = PendingMessageStore(tmp_path / "pending")
    _queue(store, ["poison", "fine"], chat_id=42)
    poison = store.items()[0]
    for _ in range(MESSAGE_MAX_ATTEMPTS - 1):  # earlier cycles already failed
        poison = store.bump_attempts(poison)
    replay = RecordingReplay(fail_texts={"poison"})
    notices: list[tuple[int, str]] = []

    _worker(
        store, FakeHealth(available=True), replay,
        notify=lambda chat_id, text: notices.append((chat_id, text)),
    )._drain_messages()

    assert notices == [(42, GAVE_UP_NOTICE.format(text="poison"))]
    # Drain continued past the poison item: "fine" was replayed and removed too.
    assert [text for _, text, _ in replay.calls] == ["poison", "fine"]
    assert store.count() == 0


def test_health_down_leaves_queue_untouched(tmp_path) -> None:
    # No point replaying into a known-down cluster: nothing is attempted and
    # attempts are not burned.
    store = PendingMessageStore(tmp_path / "pending")
    _queue(store, ["waiting"])
    replay = RecordingReplay()

    _worker(store, FakeHealth(available=False), replay)._drain_messages()

    assert replay.calls == []
    assert store.count() == 1
    assert store.items()[0].attempts == 0


# ---------------------------------------------------------------------------
# PART B — build_memory_backfill over a real day file
# ---------------------------------------------------------------------------


@dataclass
class FakeDetection:
    """Stand-in for lib.detector.Detection (label + confidence + bbox_xyxy)."""

    label: str
    confidence: float = 0.9
    bbox_xyxy: tuple = (1, 2, 3, 4)


NOW = datetime(2026, 6, 25, 15, 0)


class RecordingAnalyze:
    """Fake structured-VLM pass; behavior is mutable between runs."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.error = error

    def __call__(self, image_bytes: bytes, labels: list[str]) -> dict:
        self.calls.append(list(labels))
        if self.error is not None:
            raise self.error
        return {
            "scene": "perched by the feeder",
            "birds": [
                {"label": label, "activity": "eating", "posture": "perched", "health": ""}
                for label in labels
            ],
        }


def _write_photo(tmp_path, name: str) -> str:
    ok, buffer = cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))
    assert ok
    path = tmp_path / name
    path.write_bytes(buffer.tobytes())
    return str(path)


def _append_outage_entry(memories, when: datetime, photo: str, label: str = "percy") -> None:
    # Simulate an outage-era record: analysis=None leaves vlm_model EMPTY (the
    # durable "the VLM never ran" marker the backfill scans for).
    obs = build_observation(
        [FakeDetection(label)],
        camera="Big Cage",
        photo=photo,
        analysis=None,
        detector_model="live-019.pt",
        vlm_model="qwen2.5vl:7b",  # must NOT be stamped when analysis is None
    )
    append_entry(
        memories,
        MemoryEntry(time=when, birds=[label], note="Percy seen", photos=[photo], observations=[obs]),
    )


def test_backfill_upgrades_outage_observation_in_place(tmp_path) -> None:
    # An undecorated observation gets the VLM decoration swapped in: vlm_model
    # stamped, activity/note filled — while the ENTRY-level fields stay
    # byte-identical (they key the md/jsonl merge; changing them duplicates).
    memories = tmp_path / "memories"
    photo = _write_photo(tmp_path, "percy.jpg")
    _append_outage_entry(memories, NOW.replace(hour=10), photo)
    jsonl = memory_jsonl_path(memories, NOW.date())
    raw_before = jsonl.read_text(encoding="utf-8").splitlines()[0]
    analyze = RecordingAnalyze()

    run = build_memory_backfill(memories, analyze, "qwen2.5vl:7b", now=lambda: NOW)

    assert run() == 1
    assert analyze.calls == [["percy"]]
    records = load_day_records(memories, NOW.date())
    assert len(records) == 1
    obs = records[0]["observations"][0]
    assert obs["vlm_model"] == "qwen2.5vl:7b"
    assert obs["note"] == "perched by the feeder"
    assert obs["detections"][0]["label"] == "percy"
    assert obs["detections"][0]["activity"] == "eating"
    assert obs["detections"][0]["bbox"] == [1, 2, 3, 4]  # stored boxes kept, YOLO not re-run
    # Entry-level time/birds/note/photos are byte-identical to before the
    # rewrite: everything up to the observations array is unchanged.
    raw_after = jsonl.read_text(encoding="utf-8").splitlines()[0]
    assert raw_after.split('"observations"')[0] == raw_before.split('"observations"')[0]
    # And the merge key still matches the .md entry: exactly ONE entry on read.
    entries = load_entries(memories, NOW.date())
    assert len(entries) == 1
    assert entries[0].observations[0].detections[0].activity == "eating"


def test_backfill_pauses_on_ollama_outage_and_recovers_next_run(tmp_path) -> None:
    # Ollama dropping mid-backfill is NOT the photo's fault: the pass stops
    # without parking it and a later run (cluster back) succeeds.
    memories = tmp_path / "memories"
    photo = _write_photo(tmp_path, "percy.jpg")
    _append_outage_entry(memories, NOW.replace(hour=10), photo)
    jsonl = memory_jsonl_path(memories, NOW.date())
    before = jsonl.read_bytes()
    analyze = RecordingAnalyze(error=OllamaUnavailableError("cluster down"))

    run = build_memory_backfill(memories, analyze, "qwen2.5vl:7b", now=lambda: NOW)

    assert run() == 0
    assert jsonl.read_bytes() == before  # nothing written
    analyze.error = None  # the cluster comes back
    assert run() == 1  # not parked — retried and upgraded
    obs = load_day_records(memories, NOW.date())[0]["observations"][0]
    assert obs["vlm_model"] == "qwen2.5vl:7b"


def test_photo_parked_after_repeated_analyze_failures(tmp_path) -> None:
    # A photo the VLM keeps choking on is retried PHOTO_MAX_ATTEMPTS times and
    # then parked for this process run: analyze is no longer called for it.
    memories = tmp_path / "memories"
    photo = _write_photo(tmp_path, "percy.jpg")
    _append_outage_entry(memories, NOW.replace(hour=10), photo)
    analyze = RecordingAnalyze(error=ValueError("garbled VLM output"))

    run = build_memory_backfill(memories, analyze, "qwen2.5vl:7b", now=lambda: NOW)

    for _ in range(PHOTO_MAX_ATTEMPTS):
        assert run() == 0
    assert len(analyze.calls) == PHOTO_MAX_ATTEMPTS
    analyze.error = None  # even a now-working VLM isn't consulted: it's parked
    assert run() == 0
    assert len(analyze.calls) == PHOTO_MAX_ATTEMPTS


def test_missing_photo_is_parked_immediately(tmp_path) -> None:
    # No photo on disk means nothing to analyze — parked on first sight, not
    # retried every cycle (even if the file shows up later this process run).
    memories = tmp_path / "memories"
    missing = str(tmp_path / "gone.jpg")
    _append_outage_entry(memories, NOW.replace(hour=10), missing)
    analyze = RecordingAnalyze()

    run = build_memory_backfill(memories, analyze, "qwen2.5vl:7b", now=lambda: NOW)

    assert run() == 0
    assert analyze.calls == []
    _write_photo(tmp_path, "gone.jpg")  # appears late — already parked
    assert run() == 0
    assert analyze.calls == []


def test_batch_cap_bounds_one_run_and_next_run_finishes(tmp_path) -> None:
    # The per-cycle batch cap keeps a long backlog from hogging the cluster:
    # with 3 undecorated observations and batch=2 the first run does 2, the
    # next run picks up the remaining one.
    memories = tmp_path / "memories"
    for hour, name in ((10, "a.jpg"), (11, "b.jpg"), (12, "c.jpg")):
        _append_outage_entry(memories, NOW.replace(hour=hour), _write_photo(tmp_path, name))
    analyze = RecordingAnalyze()

    run = build_memory_backfill(memories, analyze, "qwen2.5vl:7b", now=lambda: NOW, batch=2)

    assert run() == 2
    assert run() == 1
    assert len(analyze.calls) == 3
    for record in load_day_records(memories, NOW.date()):
        assert record["observations"][0]["vlm_model"] == "qwen2.5vl:7b"


def test_already_decorated_observation_is_never_reanalyzed(tmp_path) -> None:
    # An observation that already carries vlm_model is done — re-running the
    # VLM on it would burn cluster time and could overwrite a good analysis.
    memories = tmp_path / "memories"
    photo = _write_photo(tmp_path, "percy.jpg")
    obs = build_observation(
        [FakeDetection("percy")],
        camera="Big Cage",
        photo=photo,
        analysis={"scene": "napping", "birds": []},
        detector_model="live-019.pt",
        vlm_model="qwen2.5vl:3b",
    )
    append_entry(
        memories,
        MemoryEntry(
            time=NOW.replace(hour=10), birds=["percy"], note="Percy seen",
            photos=[photo], observations=[obs],
        ),
    )
    before = memory_jsonl_path(memories, NOW.date()).read_bytes()
    analyze = RecordingAnalyze()

    run = build_memory_backfill(memories, analyze, "qwen2.5vl:7b", now=lambda: NOW)

    assert run() == 0
    assert analyze.calls == []
    assert memory_jsonl_path(memories, NOW.date()).read_bytes() == before


def test_transport_failure_pauses_drain_without_burning_attempts(tmp_path) -> None:
    # An outage mid-drain (gated call, refused connection, busy timeout) is not
    # the message's fault: the item keeps attempts=0 — the give-up counter is
    # reserved for poison messages that fail while the backend is UP.
    store = PendingMessageStore(tmp_path / "pending")
    _queue(store, ["victim", "later"])

    def replay(chat_id, text, created):
        raise OllamaUnavailableError("cluster dropped mid-drain")

    _worker(store, FakeHealth(available=True), replay)._drain_messages()

    items = store.items()
    assert [i.text for i in items] == ["victim", "later"]  # both retained
    assert [i.attempts for i in items] == [0, 0]  # no attempt burned


def test_drain_waits_until_classify_model_is_served(tmp_path) -> None:
    # Partial recovery (a worker is up but doesn't serve the classify model):
    # replaying would 404 and burn give-up attempts, so the drain must wait.
    store = PendingMessageStore(tmp_path / "pending")
    _queue(store, ["waiting"])
    replay = RecordingReplay()
    worker = LlmBackfillWorker(
        store, FakeHealth(available=True, served={"someother:7b"}),
        threading.Event(), replay=replay, llm_model="gemma3:4b",
    )

    worker._drain_messages()
    assert replay.calls == [] and store.items()[0].attempts == 0

    # Once the model appears on the cluster, the same worker drains normally.
    worker._health.served = {"gemma3:4b", "someother:7b"}
    worker._drain_messages()
    assert [t for _, t, _ in replay.calls] == ["waiting"] and store.count() == 0
