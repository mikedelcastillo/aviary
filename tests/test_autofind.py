from __future__ import annotations

import threading

from lib.autofind import AutoFinder


class FakeReg:
    def __init__(self, rows):
        self.rows = rows

    def snapshot(self):
        return list(self.rows)


class FakeFinder:
    def __init__(self):
        self.searching = False
        self.starts = []

    def is_searching(self):
        return self.searching

    def start(self, chat, target, stop):
        self.starts.append((chat, target))
        return "ok"


class FakeControl:
    def __init__(self, paused=False):
        self._p = paused

    def is_paused(self):
        return self._p


class FakeNotifier:
    user_ids = ["A"]

    def __init__(self):
        self.texts = []

    def broadcast_text(self, t):
        self.texts.append(t)

    def send_text(self, chat_id, t):
        self.texts.append(t)


def row(label, camera="cam-1", since=10.0):
    return {"label": label, "camera": camera, "since": since}


def _af(reg, finder, notifier, ir=frozenset(), total=1, control=None):
    return AutoFinder(
        reg, finder, control or FakeControl(), notifier, threading.Event(),
        ir_cameras=lambda: set(ir), camera_count=lambda: total,
        known_birds=["percy", "draft"], unseen_seconds=600, poll_seconds=1,
    )


def test_overdue_birds_searched_as_one_batch() -> None:
    reg = FakeReg([row("percy", since=700), row("draft", since=900)])
    finder, notifier = FakeFinder(), FakeNotifier()
    af = _af(reg, finder, notifier)
    af._enabled = True
    af._last_condition = "clear"
    af._tick()
    assert len(finder.starts) == 1  # ONE batched search, not two
    _, target = finder.starts[0]
    assert "percy" in target and "draft" in target
    assert any("Auto-find" in t for t in notifier.texts)


def test_recently_seen_bird_is_not_overdue() -> None:
    finder = FakeFinder()
    af = _af(FakeReg([row("percy", since=100)]), finder, FakeNotifier())
    af._enabled = True
    af._last_condition = "clear"
    af._tick()
    assert not finder.starts


def test_any_ir_camera_gates_the_search() -> None:
    finder = FakeFinder()
    af = _af(FakeReg([row("percy", since=700)]), finder, FakeNotifier(), ir={"cam-1"}, total=2)
    af._enabled = True
    af._last_condition = "mixed"  # one IR of two -> mixed, no toggle
    af._tick()
    assert not finder.starts


def test_in_progress_search_is_not_doubled() -> None:
    finder = FakeFinder()
    finder.searching = True
    af = _af(FakeReg([row("percy", since=700)]), finder, FakeNotifier())
    af._enabled = True
    af._last_condition = "clear"
    af._tick()
    assert not finder.starts


def test_auto_disables_when_all_ir_and_enables_when_clear() -> None:
    notifier = FakeNotifier()
    af = _af(FakeReg([]), FakeFinder(), notifier, ir=set(), total=2)
    af._tick()  # first tick, all clear -> enabled silently
    assert af.enabled
    af._ir_cameras = lambda: {"cam-1", "cam-2"}
    af._camera_count = lambda: 2
    af._tick()  # all IR -> auto-disable + announce
    assert not af.enabled
    assert any("night" in t.lower() or "ir" in t.lower() for t in notifier.texts)
    af._ir_cameras = lambda: set()
    af._tick()  # back to clear -> auto-enable + announce
    assert af.enabled


def test_overdue_not_repeated_until_bird_returns() -> None:
    reg = FakeReg([row("percy", since=700)])
    finder = FakeFinder()
    af = _af(reg, finder, FakeNotifier())
    af._enabled = True
    af._last_condition = "clear"
    af._tick()
    assert len(finder.starts) == 1
    af._tick()
    assert len(finder.starts) == 1  # still missing, already alerted -> no repeat
    reg.rows = [row("percy", since=10)]
    af._tick()  # percy came back
    reg.rows = [row("percy", since=700)]
    af._tick()  # missing again -> alerts a second time
    assert len(finder.starts) == 2


def test_alerted_cleared_at_dawn() -> None:
    af = _af(FakeReg([]), FakeFinder(), FakeNotifier(), ir=set(), total=2)
    af._alerted = {"percy"}
    af._last_condition = "all_ir"  # it was night
    af._auto_toggle(0, 2)          # dawn: all cameras back to daylight
    assert af._alerted == set()    # yesterday's suppression is cleared
