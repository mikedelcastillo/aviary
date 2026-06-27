from __future__ import annotations

import json
import threading

import pytest
import requests

from lib.detector import Detection
from lib.telegram.notifier import (
    MEDIA_GROUP_LIMIT,
    RETRY_AFTER_BUFFER_SECONDS,
    TelegramNotifier,
)


def detection(label: str = "bird") -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=(0, 0, 20, 20))


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


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _photo_ok(file_id: str = "BIG") -> FakeResponse:
    return FakeResponse(
        200,
        {
            "ok": True,
            "result": {
                "photo": [
                    {"file_id": "SMALL", "file_size": 100},
                    {"file_id": file_id, "file_size": 5000},
                ]
            },
        },
    )


def test_photo_uploaded_once_then_reused_by_file_id(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snap.jpg"
    snapshot.write_bytes(b"\xff\xd8image-bytes")
    post = RecordingPost([_photo_ok("BIG")])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A", "B", "C"], min_send_interval_seconds=0.0)
    notifier.send_detections("camera-1", [detection()], snapshot)

    # Exactly one call uploads the bytes (multipart `files`); the other two reuse
    # the returned file_id with no upload.
    uploads = [call for call in post.calls if "files" in call]
    reuses = [call for call in post.calls if "files" not in call]
    assert len(uploads) == 1
    assert len(reuses) == 2

    # The reuses target the remaining recipients and pass the *largest* file_id.
    assert {call["data"]["chat_id"] for call in reuses} == {"B", "C"}
    assert all(call["data"]["photo"] == "BIG" for call in reuses)
    # The upload recipient gets the real photo (no file_id reference).
    assert uploads[0]["data"]["chat_id"] == "A"
    assert "photo" not in uploads[0]["data"]


def test_broadcast_text_sends_to_every_recipient(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A", "B"], min_send_interval_seconds=0.0)
    notifier.broadcast_text("🟢 started")

    assert {call["json"]["chat_id"] for call in post.calls} == {"A", "B"}
    assert all(call["json"]["text"] == "🟢 started" for call in post.calls)
    assert all(call["url"].endswith("/sendMessage") for call in post.calls)


def test_broadcast_text_without_recipients_sends_nothing(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", [], min_send_interval_seconds=0.0)
    notifier.broadcast_text("nobody to tell")

    assert post.calls == []


def test_no_photo_sends_text_to_every_recipient(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A", "B"], min_send_interval_seconds=0.0)
    notifier.send_detections("camera-1", [detection("bird"), detection("cat")], None)

    assert len(post.calls) == 2
    assert all(call["url"].endswith("/sendMessage") for call in post.calls)
    assert {call["json"]["chat_id"] for call in post.calls} == {"A", "B"}
    # Labels are deduped, sorted and Capitalised into the message text.
    assert all(call["json"]["text"] == "Bird, Cat" for call in post.calls)


def test_text_is_sent_as_html_with_bold(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.broadcast_text("🌙 **Last night** was calm")

    call = post.calls[0]
    assert call["json"]["parse_mode"] == "HTML"
    assert call["json"]["text"] == "🌙 <b>Last night</b> was calm"


def test_send_text_returns_message_id(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True, "result": {"message_id": 99}})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)

    assert notifier.send_text("chat", "hello") == 99


def test_html_parse_error_falls_back_to_plain(monkeypatch) -> None:
    # A 400 "can't parse entities" resends as plain text with markers stripped,
    # so the message is never dropped over formatting.
    parse_400 = FakeResponse(400, {"ok": False, "description": "Bad Request: can't parse entities"})
    post = RecordingPost([parse_400, FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.broadcast_text("**hi** there")

    assert len(post.calls) == 2
    assert post.calls[0]["json"]["parse_mode"] == "HTML"
    plain = post.calls[1]["json"]
    assert "parse_mode" not in plain
    assert plain["text"] == "hi there"  # markers gone, no tags


def test_non_parse_400_is_not_retried_as_plain(monkeypatch) -> None:
    # A non-formatting 400 (e.g. chat not found) must NOT be silently resent —
    # it surfaces as an error rather than wasting a round-trip.
    bad = FakeResponse(400, {"ok": False, "description": "Bad Request: chat not found"})
    post = RecordingPost([bad])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.broadcast_text("hello")  # broadcast swallows the surfaced error

    assert len(post.calls) == 1  # only the HTML attempt, no plain resend


def test_edit_not_modified_is_a_noop_not_a_plain_resend(monkeypatch) -> None:
    # Editing to identical content 400s "message is not modified"; that's a no-op,
    # not a parse failure, so we must NOT resend a (downgraded) plain version.
    not_mod = FakeResponse(400, {"ok": False, "description": "Bad Request: message is not modified"})
    post = RecordingPost([not_mod])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    assert notifier.edit_message_text("A", 5, "**unchanged**") is True
    assert len(post.calls) == 1  # no plain resend that would drop the formatting


def test_429_pauses_then_retries(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snap.jpg"
    snapshot.write_bytes(b"bytes")
    clock = FakeClock()
    monkeypatch.setattr("lib.telegram.notifier.time.monotonic", clock.monotonic)
    monkeypatch.setattr("lib.telegram.notifier.time.sleep", clock.sleep)

    throttled = FakeResponse(429, {"ok": False, "parameters": {"retry_after": 2}})
    post = RecordingPost([throttled, _photo_ok("BIG")])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.send_detections("camera-1", [detection()], snapshot)

    # It retried after the 429 and the retry succeeded.
    assert len(post.calls) == 2
    # It paused for retry_after (2s) plus the safety buffer before retrying.
    assert clock.slept == [2.0 + RETRY_AFTER_BUFFER_SECONDS]
    # The gate stays parked until the window clears.
    assert notifier._blocked_until == pytest.approx(1000.0 + 2.0 + RETRY_AFTER_BUFFER_SECONDS)


def test_429_gives_up_after_max_retries(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr("lib.telegram.notifier.time.monotonic", clock.monotonic)
    monkeypatch.setattr("lib.telegram.notifier.time.sleep", clock.sleep)

    throttled = FakeResponse(429, {"ok": False, "parameters": {"retry_after": 1}})
    post = RecordingPost([throttled])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier(
        "token", ["A"], min_send_interval_seconds=0.0, max_send_retries=3
    )
    # Text path; the per-recipient failure is swallowed, never raised.
    notifier.send_detections("camera-1", [detection()], None)

    # Initial attempt + 3 retries = 4 posts, then it stops hammering.
    assert len(post.calls) == 4
    assert len(clock.slept) == 3


def test_send_album_uses_media_group_for_multiple(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.send_album(
        12345,
        [(b"img-a", "camera-1"), (b"img-b", "camera-2"), (b"img-c", None)],
    )

    # One sendMediaGroup call carrying all three photos in a single album.
    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"].endswith("/sendMediaGroup")
    assert call["data"]["chat_id"] == 12345

    media = json.loads(call["data"]["media"])
    assert [m["type"] for m in media] == ["photo", "photo", "photo"]
    # Each entry references its attached file by key, and captions pass through
    # (the third, None, carries no caption key).
    assert [m["media"] for m in media] == [
        "attach://photo0",
        "attach://photo1",
        "attach://photo2",
    ]
    assert media[0]["caption"] == "camera-1"
    assert media[1]["caption"] == "camera-2"
    assert "caption" not in media[2]
    # The three image payloads are attached under the matching keys.
    assert set(call["files"]) == {"photo0", "photo1", "photo2"}
    assert call["files"]["photo0"][1] == b"img-a"


def test_send_album_single_image_uses_send_photo(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.send_album(999, [(b"only", "camera-1")])

    # A lone image can't be a media group, so it ships as a plain photo upload.
    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"].endswith("/sendPhoto")
    assert call["data"] == {"chat_id": 999, "caption": "camera-1"}
    assert call["files"]["photo"][1] == b"only"


def test_send_album_chunks_beyond_media_group_limit(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    count = MEDIA_GROUP_LIMIT + 2
    notifier.send_album(7, [(f"img{i}".encode(), f"camera-{i}") for i in range(count)])

    # Split into a full album then the remaining two.
    assert len(post.calls) == 2
    first, second = post.calls
    assert len(json.loads(first["data"]["media"])) == MEDIA_GROUP_LIMIT
    assert len(json.loads(second["data"]["media"])) == 2


def test_send_album_swallows_per_chunk_failure(monkeypatch) -> None:
    # A media-group send that raises must not propagate past send_album.
    post = RecordingPost([FakeResponse(500, {"ok": False})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.send_album(1, [(b"a", "camera-1"), (b"b", "camera-2")])  # no raise

    assert len(post.calls) == 1


def test_send_album_continues_to_later_chunk_after_one_fails(monkeypatch) -> None:
    # The load-bearing promise: a failing chunk is swallowed AND the loop keeps
    # going, so a partial album still reaches the user. First album (10 photos)
    # 500s; the trailing single-photo chunk must still post.
    post = RecordingPost([FakeResponse(500, {"ok": False}), FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    count = MEDIA_GROUP_LIMIT + 1
    notifier.send_album(5, [(f"img{i}".encode(), f"camera-{i}") for i in range(count)])

    assert len(post.calls) == 2
    first, second = post.calls
    assert first["url"].endswith("/sendMediaGroup")  # the chunk that failed
    # The second chunk still shipped (a lone image => sendPhoto) with its bytes.
    assert second["url"].endswith("/sendPhoto")
    assert second["files"]["photo"][1] == f"img{MEDIA_GROUP_LIMIT}".encode()


def test_send_album_empty_sends_nothing(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.send_album(1, [])

    assert post.calls == []


def test_retry_after_falls_back_to_header(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr("lib.telegram.notifier.time.monotonic", clock.monotonic)
    monkeypatch.setattr("lib.telegram.notifier.time.sleep", clock.sleep)

    throttled = FakeResponse(429, {"ok": False}, headers={"Retry-After": "4"})
    post = RecordingPost([throttled, FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A"], min_send_interval_seconds=0.0)
    notifier.send_detections("camera-1", [detection()], None)

    assert clock.slept == [4.0 + RETRY_AFTER_BUFFER_SECONDS]


def test_broadcast_album_tracked_downscales_each_image_once(monkeypatch) -> None:
    # The digest goes to every recipient; downscaling must happen once per image,
    # not once per (image, recipient) — otherwise N recipients pay N re-encodes.
    calls = {"n": 0}

    def counting_downscale(image: bytes) -> bytes:
        calls["n"] += 1
        return image

    monkeypatch.setattr("lib.telegram.notifier.downscale_jpeg", counting_downscale)
    post = RecordingPost([FakeResponse(200, {"result": [{"message_id": 7}]})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A", "B", "C"], min_send_interval_seconds=0.0)
    sent = notifier.broadcast_album_tracked([(b"img-a", "cap"), (b"img-b", None)])

    assert calls["n"] == 2  # two images, downscaled once each (not 2 * 3 recipients)
    assert sent == {"A": 7, "B": 7, "C": 7}


def test_broadcast_album_downscales_each_image_once(monkeypatch) -> None:
    calls = {"n": 0}

    def counting_downscale(image: bytes) -> bytes:
        calls["n"] += 1
        return image

    monkeypatch.setattr("lib.telegram.notifier.downscale_jpeg", counting_downscale)
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A", "B", "C"], min_send_interval_seconds=0.0)
    notifier.broadcast_album([(b"img-a", "cap"), (b"img-b", None)])

    assert calls["n"] == 2  # downscaled once per image across all three recipients
