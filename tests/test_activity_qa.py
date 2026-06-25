from __future__ import annotations

from datetime import datetime

from lib.activity_qa import ActivityResponder, parse_activity_arg
from lib.journal import MemoryEntry, append_entry

KNOWN = ["bambi", "draft", "matcha", "percy", "pizza"]


class FakeClient:
    def chat(self, model, messages, **kwargs):
        return "Percy preened on the perch while Matcha napped nearby."


def _responder(memories_dir, notify, find=None, send_album=None, now=None):
    # Fix "now" so the window is deterministic.
    when = now or datetime(2026, 6, 25, 15, 0)
    return ActivityResponder(
        memories_dir,
        FakeClient(),
        "qwen3:4b",
        lambda: KNOWN,
        notify=notify,
        send_album=send_album,
        find=find,
        now=lambda: when,
    )


def test_parse_activity_arg() -> None:
    assert parse_activity_arg("percy today") == ("percy", True)
    assert parse_activity_arg("today") == ("", True)
    assert parse_activity_arg("percy") == ("percy", False)
    assert parse_activity_arg("") == ("", False)


def test_respond_summarises_recent_memory(tmp_path) -> None:
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"\xff\xd8jpeg")
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 40), ["percy", "matcha"], "Percy preens.", [str(photo)]))
    albums: list = []
    _responder(tmp_path, lambda c, t: None, send_album=lambda c, items: albums.append(items)).respond(
        7, "/activity percy", "percy"
    )
    # One album sent, summary is the first item's caption, photo included.
    assert len(albums) == 1
    items = albums[0]
    assert len(items) == 1
    assert "preened" in items[0][1]


def test_respond_today_window(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["bambi"], "Bambi ate early."))
    sent: list = []
    # "today" -> since midnight, so the 08:00 entry is included even though it's
    # well outside the last hour.
    _responder(tmp_path, lambda c, t: sent.append(t)).respond(7, "/activity bambi today", "bambi today")
    assert sent  # a summary was produced from the morning entry


def test_respond_triggers_find_when_unlogged_and_live(tmp_path) -> None:
    found: list = []
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: found.append((cid, arg))).respond(
        7, "what is draft doing right now?", "draft"
    )
    assert found == [(7, "draft")]
    assert any("check the cameras" in t for t in sent)


def test_respond_says_unlogged_when_not_live(tmp_path) -> None:
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: None).respond(
        7, "/activity matcha", "matcha"
    )
    assert sent and "haven't logged" in sent[0].lower()
