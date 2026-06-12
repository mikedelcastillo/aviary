from __future__ import annotations

import threading

import pytest
import requests

from lib.detector import Detection
from lib.telegram.notifier import RETRY_AFTER_BUFFER_SECONDS, TelegramNotifier


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


def test_no_photo_sends_text_to_every_recipient(monkeypatch) -> None:
    post = RecordingPost([FakeResponse(200, {"ok": True})])
    monkeypatch.setattr("lib.telegram.notifier.requests.post", post)

    notifier = TelegramNotifier("token", ["A", "B"], min_send_interval_seconds=0.0)
    notifier.send_detections("camera-1", [detection("bird"), detection("cat")], None)

    assert len(post.calls) == 2
    assert all(call["url"].endswith("/sendMessage") for call in post.calls)
    assert {call["json"]["chat_id"] for call in post.calls} == {"A", "B"}
    # Labels are deduped and sorted into the message text.
    assert all(call["json"]["text"] == "bird, cat" for call in post.calls)


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
