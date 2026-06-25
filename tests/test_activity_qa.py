from __future__ import annotations

from datetime import datetime

from lib.activity_qa import ActivityResponder, parse_activity_arg
from lib.journal import MemoryEntry, append_entry

KNOWN = ["bambi", "draft", "matcha", "percy", "pizza"]


class FakeClient:
    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    def chat(self, model, messages, **kwargs):
        self.system = messages[0]["content"]
        self.user = messages[1]["content"]
        return "Percy preened on the perch while Matcha napped nearby."


def _responder(memories_dir, notify, find=None, send_album=None, now=None, client=None, care_answer=None):
    # Fix "now" so the window is deterministic.
    when = now or datetime(2026, 6, 25, 15, 0)
    return ActivityResponder(
        memories_dir,
        client or FakeClient(),
        "qwen3:4b",
        lambda: KNOWN,
        notify=notify,
        send_album=send_album,
        find=find,
        pronoun_note="Percy and Bambi are female (use she/her for them).",
        now=lambda: when,
        care_answer=care_answer,
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


def test_care_fallback_answers_unlogged_care_question(tmp_path) -> None:
    # A care question routed to the activity path with no logged memory should be
    # answered from care knowledge, not dead-end on "I haven't logged that".
    sent: list = []
    asked: list = []

    def care_answer(text: str) -> str:
        asked.append(text)
        return "Keep them at a steady 65–80°F and out of drafts."

    _responder(tmp_path, lambda c, t: sent.append(t), care_answer=care_answer).respond(
        7, "is it too cold for percy", "percy"
    )
    assert asked == ["is it too cold for percy"]
    assert any("65–80" in s for s in sent)
    assert not any("haven't logged" in s.lower() for s in sent)


def test_care_question_with_now_answers_care_not_camera_search(tmp_path) -> None:
    # A care question carrying a live word ("now") must be answered from care
    # knowledge, NOT sent off to a camera search before the care fallback runs.
    sent: list = []
    found: list = []

    def care_answer(text: str):
        return "Keep them at 65–80°F." if "cold" in text else None

    _responder(
        tmp_path,
        lambda c, t: sent.append(t),
        find=lambda cid, arg: found.append(arg),
        care_answer=care_answer,
    ).respond(7, "is it too cold for percy now", "percy")
    assert any("65–80" in s for s in sent)
    assert found == []  # not handed to the cameras


def test_question_uses_qa_path_with_birds_and_question(tmp_path) -> None:
    # Whole-day co-occurrence question -> QA prompt, notes carry each entry's birds.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 0), ["jynx", "matcha"], "Together on the perch."))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 13, 0), ["pizza"], "Pizza alone."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "did jynx and matcha spend time together today?", "jynx and matcha"
    )
    # The QA system prompt (not the bullet-summary one) was used.
    assert "answering a question" in client.system.lower()
    # The user payload includes the question and the per-entry bird list.
    assert "spend time together" in client.user
    assert "Jynx, Matcha" in client.user


def test_question_defaults_to_whole_day_window(tmp_path) -> None:
    # "did pizza eat" (no "today") still looks at the whole day, not just last hour.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 30), ["pizza"], "Pizza at the food bowl."))
    client = FakeClient()
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), client=client).respond(
        7, "did pizza eat?", "pizza"
    )
    assert "food bowl" in client.user  # the 08:30 entry was in-window
    assert "from today" in client.user


def test_morning_window_excludes_afternoon(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["bambi"], "Morning bath."))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["bambi"], "Afternoon nap."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "what did bambi do this morning?", "bambi"
    )
    assert "Morning bath" in client.user
    assert "Afternoon nap" not in client.user
    assert "this morning" in client.user


def test_together_request_prefers_multi_bird_photos(tmp_path) -> None:
    duo = tmp_path / "duo.jpg"; duo.write_bytes(b"duo-bytes")
    solo = tmp_path / "solo.jpg"; solo.write_bytes(b"solo-bytes")
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 0), ["bambi", "jynx"], "Together.", [str(duo)]))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 13, 0), ["bambi"], "Bambi alone.", [str(solo)]))
    albums: list = []
    _responder(tmp_path, lambda c, t: None, send_album=lambda c, items: albums.append(items)).respond(
        7, "show me photos of bambi with other birds", "bambi"
    )
    assert albums
    sent = [img for img, _ in albums[0]]
    assert duo.read_bytes() in sent and solo.read_bytes() not in sent


def test_week_window_includes_earlier_days(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 22, 9, 0), ["jynx", "bambi"], "Earlier this week together."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "has bambi spent time with jynx this week?", "bambi and jynx"
    )
    assert "Earlier this week" in client.user


def test_evening_window_before_5pm_falls_back_to_today(tmp_path) -> None:
    # Asked at 10am, "tonight" would be an inverted 17:00->10:00 range; must not
    # invert — fall back to the whole day.
    r = _responder(tmp_path, lambda c, t: None, now=datetime(2026, 6, 25, 10, 0))
    since, until, phrase = r._window("are the birds asleep tonight", "", datetime(2026, 6, 25, 10, 0), True)
    assert since <= until and phrase == "today"


def test_last_night_uses_previous_evening(tmp_path) -> None:
    r = _responder(tmp_path, lambda c, t: None)
    now = datetime(2026, 6, 25, 9, 0)
    since, until, phrase = r._window("what happened last night", "", now, True)
    assert phrase == "last night"
    assert since.day == 24 and since < until  # spans into the previous day


def test_wants_live_is_word_boundary() -> None:
    from lib.activity_qa import _wants_live
    assert _wants_live("what is percy doing now") is True
    assert _wants_live("do you know what percy did today") is False  # 'know', not 'now'
