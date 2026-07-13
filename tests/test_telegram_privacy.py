"""TelegramNotifier privacy behavior: photos the person screen flags are
withheld from EVERY Telegram upload path, while the message text still goes out
carrying the withheld-photo note."""

from __future__ import annotations

import json
import threading

import requests

from lib.detector import Detection
from lib.telegram.notifier import PHOTO_WITHHELD_NOTE, TelegramNotifier


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, headers=None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class RecordingPost:
    """Thread-safe stand-in for requests.post that returns queued responses."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, url, **kwargs):
        with self._lock:
            self.calls.append({"url": url, **kwargs})
            response = (
                self._responses.pop(0)
                if len(self._responses) > 1
                else self._responses[0]
            )
        return response


class FakeScreen:
    """Privacy-screen stand-in flagging exact byte payloads as 'has a person'."""

    def __init__(self, flagged: set[bytes] | None = None, error: Exception | None = None) -> None:
        self.flagged = flagged or set()
        self.error = error
        self.calls = 0

    def has_person(self, image_bytes: bytes) -> bool:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return image_bytes in self.flagged


def detection(label: str = "bird") -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=(0, 0, 20, 20))


def _notifier(user_ids, screen) -> TelegramNotifier:
    return TelegramNotifier(
        "token", user_ids, min_send_interval_seconds=0.0, person_screen=screen
    )


def _urls(post: RecordingPost) -> list[str]:
    return [call["url"].rsplit("/", 1)[-1] for call in post.calls]


# --- send_detections (alert photos) -----------------------------------------


def test_alert_photo_withheld_but_text_alert_still_broadcast(monkeypatch, tmp_path) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)
    snapshot = tmp_path / "snap.jpg"
    snapshot.write_bytes(b"\xff\xd8person-bytes")

    screen = FakeScreen(flagged={snapshot.read_bytes()})
    notifier = _notifier(["A", "B"], screen)
    notifier.send_detections("aviary", [detection("percy")], snapshot)

    assert set(_urls(post)) == {"sendMessage"}
    assert {call["json"]["chat_id"] for call in post.calls} == {"A", "B"}
    for call in post.calls:
        assert "Percy" in call["json"]["text"]
        assert PHOTO_WITHHELD_NOTE in call["json"]["text"]


def test_alert_photo_clean_is_uploaded_and_screened_once(monkeypatch, tmp_path) -> None:
    post = RecordingPost(
        [
            FakeResponse(
                200,
                {"ok": True, "result": {"photo": [{"file_id": "BIG", "file_size": 5000}]}},
            )
        ]
    )
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)
    snapshot = tmp_path / "snap.jpg"
    snapshot.write_bytes(b"\xff\xd8bird-bytes")

    screen = FakeScreen()
    notifier = _notifier(["A", "B"], screen)
    notifier.send_detections("aviary", [detection("percy")], snapshot)

    assert all(url == "sendPhoto" for url in _urls(post))
    # One screening for the whole fan-out, not one per recipient.
    assert screen.calls == 1


# --- send_photo (find proof / Q&A) -------------------------------------------


def test_withheld_photo_becomes_caption_plus_note_and_reports_success(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True, "result": {"message_id": 5}})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"me"}))
    assert notifier.send_photo("A", b"me", "🔎 Found Percy!") is True

    assert _urls(post) == ["sendMessage"]
    assert post.calls[0]["json"]["text"] == f"🔎 Found Percy!\n{PHOTO_WITHHELD_NOTE}"


def test_withheld_photo_without_caption_sends_bare_note(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"me"}))
    assert notifier.send_photo("A", b"me", None) is True
    assert post.calls[0]["json"]["text"] == PHOTO_WITHHELD_NOTE


def test_withheld_note_delivery_failure_returns_false(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(500, {"ok": False})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"me"}))
    assert notifier.send_photo("A", b"me", "caption") is False


def test_clean_photo_still_uploads(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen())
    assert notifier.send_photo("A", b"bird", "caption") is True
    assert _urls(post) == ["sendPhoto"]


# --- fail-closed -------------------------------------------------------------


def test_screen_error_withholds_photo(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(error=RuntimeError("screen died")))
    assert notifier.send_photo("A", b"who knows", "caption") is True
    assert _urls(post) == ["sendMessage"]
    assert PHOTO_WITHHELD_NOTE in post.calls[0]["json"]["text"]


def test_without_screen_photos_pass_untouched(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    assert notifier.send_photo("A", b"anything", None) is True
    assert _urls(post) == ["sendPhoto"]


# --- send_album (/snapshot, activity photos) ---------------------------------


def test_album_drops_flagged_photo_and_sends_note(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"img-b"}))
    notifier.send_album(
        "A", [(b"img-a", "cam 1"), (b"img-b", "cam 2"), (b"img-c", "cam 3")]
    )

    urls = _urls(post)
    assert urls.count("sendMediaGroup") == 1
    assert urls.count("sendMessage") == 1
    group_call = post.calls[[u == "sendMediaGroup" for u in urls].index(True)]
    media = json.loads(group_call["data"]["media"])
    assert [entry.get("caption") for entry in media] == ["cam 1", "cam 3"]
    note_call = post.calls[[u == "sendMessage" for u in urls].index(True)]
    assert note_call["json"]["text"] == PHOTO_WITHHELD_NOTE


def test_album_all_withheld_sends_note_only(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"img-a", b"img-b"}))
    notifier.send_album("A", [(b"img-a", "cam 1"), (b"img-b", "cam 2")])

    assert _urls(post) == ["sendMessage"]
    assert "2 photos withheld" in post.calls[0]["json"]["text"]


def test_album_leading_caption_moves_to_first_kept_photo(monkeypatch) -> None:
    # Caretaker/activity pattern: only item 0 carries the (summary) caption.
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"img-a"}))
    notifier.send_album("A", [(b"img-a", "🐦 the summary"), (b"img-b", None)])

    photo_calls = [call for call in post.calls if call["url"].endswith("sendPhoto")]
    assert len(photo_calls) == 1
    assert photo_calls[0]["data"]["caption"] == "🐦 the summary"


def test_album_per_photo_captions_never_move(monkeypatch) -> None:
    # /snapshot pattern: captions name cameras; a withheld photo's caption must
    # not migrate onto a different camera's photo.
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"img-a"}))
    notifier.send_album("A", [(b"img-a", "cam 1"), (b"img-b", "cam 2")])

    photo_calls = [call for call in post.calls if call["url"].endswith("sendPhoto")]
    assert len(photo_calls) == 1
    assert photo_calls[0]["data"]["caption"] == "cam 2"


# --- broadcast_album / broadcast_album_tracked --------------------------------


def test_broadcast_album_screens_once_for_all_recipients(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    screen = FakeScreen(flagged={b"img-b"})
    notifier = _notifier(["A", "B"], screen)
    notifier.broadcast_album([(b"img-a", None), (b"img-b", None)])

    assert screen.calls == 2  # once per IMAGE, not per image x recipient
    urls = _urls(post)
    assert urls.count("sendPhoto") == 2  # one kept photo to each recipient
    assert urls.count("sendMessage") == 2  # one note to each recipient


def test_tracked_album_all_withheld_notes_and_returns_empty(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A", "B"], FakeScreen(flagged={b"img-a"}))
    sent = notifier.broadcast_album_tracked([(b"img-a", "🐦 summary")])

    # Empty result lets the caller's existing text fallback deliver the report;
    # the note explains the missing photo.
    assert sent == {}
    assert set(_urls(post)) == {"sendMessage"}
    assert all(PHOTO_WITHHELD_NOTE == call["json"]["text"] for call in post.calls)


def test_tracked_album_partial_withhold_keeps_summary_and_tracking(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True, "result": {"message_id": 7}})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = _notifier(["A"], FakeScreen(flagged={b"img-a"}))
    sent = notifier.broadcast_album_tracked(
        [(b"img-a", "🐦 summary"), (b"img-b", None)]
    )

    assert sent == {"A": 7}
    photo_calls = [call for call in post.calls if call["url"].endswith("sendPhoto")]
    assert len(photo_calls) == 1
    assert photo_calls[0]["data"]["caption"] == "🐦 summary"
