from __future__ import annotations

import json
from datetime import datetime

from lib.pending import PendingMessageStore


def test_add_persists_json_and_items_round_trips(tmp_path) -> None:
    # add() must write a real .json file and items() must read back every field.
    store = PendingMessageStore(tmp_path)
    when = datetime(2026, 7, 4, 8, 30, 15, 123456)
    item = store.add(42, "is the cluster back?", when)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].stem == item.id
    loaded = store.items()
    assert len(loaded) == 1
    got = loaded[0]
    assert got.id == item.id
    assert got.chat_id == 42
    assert got.text == "is the cluster back?"
    assert got.created == when
    assert got.attempts == 0


def test_items_returns_oldest_first(tmp_path) -> None:
    # Filenames start with the timestamp, so lexicographic sort == chronological.
    store = PendingMessageStore(tmp_path)
    store.add(1, "first", datetime(2026, 7, 4, 8, 0, 0))
    store.add(1, "second", datetime(2026, 7, 4, 9, 0, 0))
    store.add(1, "third", datetime(2026, 7, 4, 10, 0, 0))
    assert [i.text for i in store.items()] == ["first", "second", "third"]


def test_same_second_adds_get_unique_ids(tmp_path) -> None:
    # Two messages in the same (microsecond-free) second must not collide:
    # the per-store seq counter differentiates the filenames.
    store = PendingMessageStore(tmp_path)
    when = datetime(2026, 7, 4, 8, 0, 0)  # microsecond == 0
    a = store.add(7, "one", when)
    b = store.add(7, "two", when)
    assert a.id != b.id
    assert store.count() == 2
    assert sorted(i.text for i in store.items()) == ["one", "two"]


def test_remove_deletes_and_count_tracks(tmp_path) -> None:
    store = PendingMessageStore(tmp_path)
    a = store.add(1, "keep", datetime(2026, 7, 4, 8, 0, 0))
    b = store.add(1, "drop", datetime(2026, 7, 4, 8, 0, 1))
    assert store.count() == 2
    store.remove(b.id)
    assert store.count() == 1
    assert [i.id for i in store.items()] == [a.id]
    # Removing a missing id is a no-op, not an error.
    store.remove(b.id)
    assert store.count() == 1


def test_bump_attempts_persists_across_restart(tmp_path) -> None:
    # attempts must be written to disk so a crash between retries doesn't
    # reset the retry count — a fresh store over the same dir sees it.
    store = PendingMessageStore(tmp_path)
    item = store.add(9, "retry me", datetime(2026, 7, 4, 8, 0, 0))
    updated = store.bump_attempts(item)
    assert updated.attempts == 1
    updated = store.bump_attempts(updated)
    assert updated.attempts == 2
    reborn = PendingMessageStore(tmp_path)  # simulated restart
    items = reborn.items()
    assert len(items) == 1
    assert items[0].attempts == 2


def test_malformed_json_file_is_skipped(tmp_path) -> None:
    # A corrupt file must not take down the whole queue drain.
    store = PendingMessageStore(tmp_path)
    good = store.add(3, "good", datetime(2026, 7, 4, 8, 0, 0))
    (tmp_path / "00000000_000000_000000_000_1.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "zzz_missing_keys.json").write_text(json.dumps({"text": "no chat_id"}), encoding="utf-8")
    items = store.items()
    assert [i.id for i in items] == [good.id]


def test_leftover_tmp_file_not_picked_up(tmp_path) -> None:
    # A crash mid-_write leaves a .json.tmp; items()/count() glob only *.json
    # so the half-written temp file is invisible to the queue.
    store = PendingMessageStore(tmp_path)
    good = store.add(5, "real", datetime(2026, 7, 4, 8, 0, 0))
    (tmp_path / "orphan.json.tmp").write_text("{", encoding="utf-8")
    assert store.count() == 1
    assert [i.id for i in store.items()] == [good.id]
