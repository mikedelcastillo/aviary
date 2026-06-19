from __future__ import annotations

import threading

from lib.detector import Detection
from lib.dashboard import _format_frame_age
from lib.objects import ObjectRegistry
from lib.stats import CameraStats
from lib.telegram.commands import build_status_message, run_command_bot


class Response:
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def detection(label: str = "bird", bbox=(0, 0, 20, 20)) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=bbox)


def test_status_message_includes_dashboard_data_without_logs() -> None:
    registry = ObjectRegistry()
    stats = {"camera-1": CameraStats("camera-1", 0.25, registry)}
    stats["camera-1"].set_status("connected")
    stats["camera-1"].record_inference(["bird"], [detection()], (100, 100))
    stats["camera-1"].record_alert()

    message = build_status_message(stats, registry, movement_alert_ratio=0.10)

    assert "Aviary status" in message
    assert "camera-1: CONNECTED" in message
    assert "1 frames, 1 detections, 1 alerts" in message
    assert "camera-1 bird:" in message
    assert "Logs" not in message
    assert "Events" not in message


def test_status_message_formats_recent_frame_age_in_milliseconds(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.stats.time.monotonic", lambda: now)
    stats = {"camera-1": CameraStats("camera-1", 0.25)}
    stats["camera-1"].set_status("connected")
    stats["camera-1"].record_inference([])
    now = 100.125

    message = build_status_message(stats, None, movement_alert_ratio=0.10)

    assert "Last frame: 125ms ago" in message


def test_dashboard_frame_age_uses_seconds_after_one_second() -> None:
    assert _format_frame_age(1.25) == "1.2s"


def test_status_command_requires_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    provider_called = False

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/status",
                            "from": {"id": 999},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def status_provider() -> str:
        nonlocal provider_called
        provider_called = True
        return "secret status"

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        status_provider=status_provider,
        stop_event=stop_event,
        poll_timeout_seconds=0,
    )

    assert sent_messages == ["Unauthorized."]
    assert provider_called is False


def test_status_command_replies_for_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/status",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        status_provider=lambda: "secret status",
        stop_event=stop_event,
        poll_timeout_seconds=0,
    )

    assert sent_messages == ["secret status"]


def test_discover_command_requires_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    provider_called = False

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/discover",
                            "from": {"id": 999},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def discover_provider() -> str:
        nonlocal provider_called
        provider_called = True
        return "discovery report"

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        discover_provider=discover_provider,
    )

    assert sent_messages == ["Unauthorized."]
    assert provider_called is False


def test_discover_command_acks_then_reports_for_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/discover",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        sent_messages.append(json["text"])
        # Stop only once both the ack and the report have been sent so the loop
        # doesn't exit before the second message.
        if len(sent_messages) >= 2:
            stop_event.set()
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        discover_provider=lambda: "found 2 cameras",
    )

    assert sent_messages == [
        "Scanning the local network for cameras...",
        "found 2 cameras",
    ]
