from __future__ import annotations

import threading
import time

from lib.detector import Detection
from lib.dashboard import _format_frame_age
from lib.objects import ObjectRegistry
from lib.stats import CameraStats
from lib.telegram.commands import (
    _register_bot_commands,
    build_status_message,
    run_command_bot,
)


class Response:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def detection(label: str = "bird", bbox=(0, 0, 20, 20)) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=bbox)


def test_status_message_shows_birds_last_seen_and_cameras() -> None:
    registry = ObjectRegistry()
    stats = {"camera-1": CameraStats("camera-1", 0.25, registry)}
    stats["camera-1"].set_status("connected")
    stats["camera-1"].record_inference(["percy"], [detection("percy")], (100, 100))

    message = build_status_message(stats, registry)

    assert "Aviary" in message
    assert "Birds — last seen" in message
    assert "Percy" in message  # the seen bird appears with its last-seen
    assert "camera-1" in message
    assert "Logs" not in message and "Events" not in message


def test_status_message_lists_missing_roster_birds_and_ir() -> None:
    registry = ObjectRegistry()
    stats = {"camera-1": CameraStats("camera-1", 0.25, registry)}
    stats["camera-1"].set_status("connected")
    stats["camera-1"].record_inference(["percy"], [detection("percy")], (100, 100))

    message = build_status_message(
        stats, registry, known_birds=["percy", "draft"], ir_cameras={"camera-1"}
    )
    # Percy seen, Draft listed as not-seen; the IR camera is marked.
    assert "Percy" in message
    assert "not seen yet" in message and "Draft" in message
    assert "🌙 IR" in message


def test_status_message_uses_logged_last_seen_for_roster_birds() -> None:
    registry = ObjectRegistry()
    stats = {"camera-1": CameraStats("camera-1", 0.25, registry)}
    stats["camera-1"].set_status("connected")

    message = build_status_message(
        stats,
        registry,
        known_birds=["jynx", "draft"],
        logged_last_seen={"jynx": (90.0, "camera-1")},
    )

    assert "Jynx — 1m 30s ago" in message
    assert "not seen yet: Draft" in message
    assert "not seen yet: Jynx" not in message


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
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
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
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
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
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
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


def test_discover_command_acks_then_edits_report_for_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    calls: list[dict] = []

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

    def post(url, json, timeout):
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
        calls.append({"url": url, "json": json})
        if json["text"] == "found 2 cameras":
            stop_event.set()
        return Response({"result": {"message_id": 77}})

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        discover_provider=lambda: "found 2 cameras",
    )

    assert [call["json"]["text"] for call in calls] == [
        "🔍 Scanning the local network for cameras…",
        "found 2 cameras",
    ]
    assert calls[0]["url"].endswith("/sendMessage")
    assert calls[1]["url"].endswith("/editMessageText")
    assert calls[1]["json"]["message_id"] == 77


def test_discover_command_edits_ack_when_message_id_available(monkeypatch) -> None:
    stop_event = threading.Event()
    calls: list[dict] = []

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

    def post(url, json, timeout):
        if "text" not in json:
            return Response()
        calls.append({"url": url, "json": json})
        if url.endswith("/editMessageText"):
            stop_event.set()
            return Response()
        return Response({"result": {"message_id": 77}})

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        discover_provider=lambda: "found 2 cameras",
    )

    assert len(calls) == 2
    assert calls[0]["url"].endswith("/sendMessage")
    assert calls[0]["json"]["text"] == "🔍 Scanning the local network for cameras…"
    assert calls[1]["url"].endswith("/editMessageText")
    assert calls[1]["json"]["message_id"] == 77
    assert calls[1]["json"]["text"] == "found 2 cameras"


def test_discover_command_edits_live_progress_before_final_report(monkeypatch) -> None:
    stop_event = threading.Event()
    progress_seen = threading.Event()
    calls: list[dict] = []

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

    def post(url, json, timeout):
        if "text" not in json:
            return Response()
        calls.append({"url": url, "json": json})
        if json["text"] == "found 1 so far":
            progress_seen.set()
        if json["text"] == "final report":
            stop_event.set()
        return Response({"result": {"message_id": 77}})

    def discover_provider(progress_update) -> str:
        # Publish a live frame, then hold the scan open until the steady-tick
        # thread has drawn it, so the intermediate edit is observed before the
        # final report (mirrors a real sweep that runs across several ticks).
        progress_update("found 1 so far")
        progress_seen.wait(2.0)
        return "final report"

    # Shrink the live-frame cadence so the tick fires immediately under test.
    monkeypatch.setattr("lib.telegram.commands.DISCOVER_TICK_SECONDS", 0.01)
    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        discover_provider=discover_provider,
    )

    assert [call["json"]["text"] for call in calls] == [
        "🔍 Scanning the local network for cameras…",
        "found 1 so far",
        "final report",
    ]
    assert calls[1]["url"].endswith("/editMessageText")
    assert calls[1]["json"]["message_id"] == 77
    assert calls[2]["url"].endswith("/editMessageText")
    assert calls[2]["json"]["message_id"] == 77


def test_restart_command_calls_provider_for_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    calls: list[bool] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/restart",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def restart_provider() -> str:
        calls.append(True)
        return "Restarting Aviary server..."

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        restart_provider=restart_provider,
    )

    assert calls == [True]
    assert sent_messages == ["Restarting Aviary server..."]


def test_detections_command_passes_argument_to_provider(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    seen_args: list[str] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/detections percy 2026-06-27",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def detection_provider(argument: str) -> str:
        seen_args.append(argument)
        return "Percy — 1h"

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        detection_provider=detection_provider,
    )

    assert seen_args == ["percy 2026-06-27"]
    assert sent_messages == ["Percy — 1h"]


def test_home_command_calls_provider_for_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    calls: list[bool] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/home",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    def home_provider() -> str:
        calls.append(True)
        return "🏠 Sent 2/2 pan-tilt camera(s) to their saved viewpoint."

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        home_provider=home_provider,
    )

    assert calls == [True]
    assert sent_messages == ["🏠 Sent 2/2 pan-tilt camera(s) to their saved viewpoint."]


def test_quality_command_passes_argument_to_provider(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    seen_args: list[str] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/quality stream2",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def quality_provider(argument: str) -> str:
        seen_args.append(argument)
        return "Quality mode: stream2."

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        quality_provider=quality_provider,
    )

    assert seen_args == ["stream2"]
    assert sent_messages == ["Quality mode: stream2."]


def test_snapshot_command_requires_allowed_user(monkeypatch) -> None:
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
                            "text": "/snapshot",
                            "from": {"id": 999},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def snapshot_provider(_chat_id: int) -> str:
        nonlocal provider_called
        provider_called = True
        return "album sent"

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        snapshot_provider=snapshot_provider,
    )

    assert sent_messages == ["Unauthorized."]
    assert provider_called is False


def test_register_bot_commands_skips_unknown_command(monkeypatch) -> None:
    posted: list[dict] = []

    def post(_url, json, timeout):
        posted.append(json)
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    # A command with no menu description is a wiring mistake, not a Telegram API
    # failure. It must be skipped so the rest of the menu still registers and the
    # bot still starts — never crash startup with a KeyError.
    _register_bot_commands("https://api.telegram.org/bottoken", ["/status", "/bogus"])

    assert len(posted) == 1
    assert [entry["command"] for entry in posted[0]["commands"]] == ["status"]


def test_register_bot_commands_sends_nothing_when_all_unknown(monkeypatch) -> None:
    posted: list[dict] = []

    def post(_url, json, timeout):
        posted.append(json)
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    # An empty payload would CLEAR the bot's menu, so skip the call entirely.
    _register_bot_commands("https://api.telegram.org/bottoken", ["/bogus"])

    assert posted == []


def test_registers_available_bot_commands_on_startup(monkeypatch) -> None:
    stop_event = threading.Event()
    registered: list[dict] = []

    def get(_url, params, timeout):
        # First poll has nothing to handle; stop so the loop exits after the
        # startup registration has already run.
        stop_event.set()
        return Response({"result": []})

    def post(url, json, timeout):
        if url.endswith("/setMyCommands"):
            registered.extend(json["commands"])
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        status_provider=lambda: "status",
        stop_event=stop_event,
        poll_timeout_seconds=0,
        discover_provider=lambda: "report",
        snapshot_provider=lambda _chat_id: "album",
    )

    # Every wired-up command is offered to Telegram's slash-menu, in a stable
    # display order, as bare names (no leading slash) with non-empty descriptions.
    assert [entry["command"] for entry in registered] == [
        "discover",
        "status",
        "snapshot",
        "userinfo",
    ]
    assert all(entry["description"] for entry in registered)
    assert all(not entry["command"].startswith("/") for entry in registered)


def test_registers_only_userinfo_when_no_providers(monkeypatch) -> None:
    stop_event = threading.Event()
    registered: list[dict] = []

    def get(_url, params, timeout):
        stop_event.set()
        return Response({"result": []})

    def post(url, json, timeout):
        if url.endswith("/setMyCommands"):
            registered.extend(json["commands"])
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    # user_id mode: no allowed users and no providers, so only /userinfo works
    # and only /userinfo should appear in the autocomplete menu.
    run_command_bot(
        "token",
        allowed_user_ids=[],
        stop_event=stop_event,
        poll_timeout_seconds=0,
    )

    assert [entry["command"] for entry in registered] == ["userinfo"]


def test_snapshot_command_acks_then_reports_with_chat_id(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    seen_chat_ids: list[int] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/snapshot",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        # Stop only once both the ack and the report have been sent.
        if len(sent_messages) >= 2:
            stop_event.set()
        return Response()

    def snapshot_provider(chat_id: int) -> str:
        seen_chat_ids.append(chat_id)
        return "Sent 3 snapshot(s)."

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        snapshot_provider=snapshot_provider,
    )

    assert sent_messages == [
        "Capturing snapshots from all cameras...",
        "Sent 3 snapshot(s).",
    ]
    # The requesting chat is handed to the provider so the album goes to them.
    assert seen_chat_ids == [123]


def _single_update(text: str, user_id: int = 111, chat_id: int = 123):
    """A getUpdates stub yielding one message with ``text``."""

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": text,
                            "from": {"id": user_id},
                            "chat": {"id": chat_id},
                        },
                    }
                ]
            }
        )

    return get


def test_pause_command_parses_duration_and_calls_provider(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    seen_durations: list[float | None] = []

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def pause_provider(duration: float | None) -> str:
        seen_durations.append(duration)
        return "Privacy mode on."

    monkeypatch.setattr("lib.telegram.commands.requests.get", _single_update("/pause 10m"))
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        pause_provider=pause_provider,
    )

    # "10m" -> 600 seconds handed to the provider; its reply is forwarded.
    assert seen_durations == [600.0]
    assert sent_messages == ["Privacy mode on."]


def test_pause_command_without_argument_is_indefinite(monkeypatch) -> None:
    stop_event = threading.Event()
    seen_durations: list[float | None] = []

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        stop_event.set()
        return Response()

    def pause_provider(duration: float | None) -> str:
        seen_durations.append(duration)
        return "Paused indefinitely."

    monkeypatch.setattr("lib.telegram.commands.requests.get", _single_update("/stop"))
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        pause_provider=pause_provider,
    )

    # No argument -> None -> indefinite pause. /stop is an alias of /pause.
    assert seen_durations == [None]


def test_pause_command_requires_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    provider_called = False

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def pause_provider(_duration: float | None) -> str:
        nonlocal provider_called
        provider_called = True
        return "Privacy mode on."

    monkeypatch.setattr(
        "lib.telegram.commands.requests.get", _single_update("/pause", user_id=999)
    )
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        pause_provider=pause_provider,
    )

    assert sent_messages == ["Unauthorized."]
    assert provider_called is False


def test_resume_command_calls_provider(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    provider_called = False

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    def resume_provider() -> str:
        nonlocal provider_called
        provider_called = True
        return "Resumed — cameras are live again."

    monkeypatch.setattr("lib.telegram.commands.requests.get", _single_update("/play"))
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        resume_provider=resume_provider,
    )

    assert provider_called is True
    assert sent_messages == ["Resumed — cameras are live again."]


def test_photo_message_runs_photo_provider(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    seen_image: list[bytes] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "photo": [
                                {"file_id": "small"},
                                {"file_id": "BIG"},  # largest last
                            ],
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        # Stop once both the ack and the analysis reply have been sent.
        if len(sent_messages) >= 2:
            stop_event.set()
        return Response()

    def photo_provider(image: bytes) -> str:
        seen_image.append(image)
        return "📸 I spotted: Percy!"

    # download_telegram_file is exercised separately; stub it here.
    monkeypatch.setattr(
        "lib.telegram.commands.download_telegram_file", lambda base, fid, timeout=30: b"jpeg-" + fid.encode()
    )
    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        photo_provider=photo_provider,
    )

    # The LARGEST photo is downloaded and handed to the provider; its reply ships.
    assert seen_image == [b"jpeg-BIG"]
    assert sent_messages == ["📷 Taking a look at your photo…", "📸 I spotted: Percy!"]


def test_status_shows_ir_species_label_during_night() -> None:
    registry = ObjectRegistry()
    stats = {"camera-1": CameraStats("camera-1", 0.25, registry)}
    stats["camera-1"].set_status("connected")
    # At night only the species/IR outline is detected, not the individual.
    stats["camera-1"].record_inference(["cockatiel"], [detection("cockatiel")], (100, 100))
    message = build_status_message(stats, registry, known_birds=["percy", "draft"])
    assert "Cockatiel" in message  # the seen species shows, not just "nothing seen"


def test_sleep_command_passes_argument_to_provider(monkeypatch) -> None:
    from lib.telegram.commands import COMMAND_DESCRIPTIONS

    assert "/sleep" in COMMAND_DESCRIPTIONS  # registered in the slash menu

    stop_event = threading.Event()
    seen: list[tuple[int, str]] = []

    def post(_url, json, timeout):
        return Response()

    def sleep_provider(chat_id: int, argument: str) -> None:
        seen.append((chat_id, argument))
        stop_event.set()

    monkeypatch.setattr("lib.telegram.commands.requests.get", _single_update("/sleep week"))
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        sleep_provider=sleep_provider,
    )

    assert seen == [(123, "week")]


def test_command_bot_survives_a_handler_exception(monkeypatch) -> None:
    """A handler raising must not kill the poll thread; the bot keeps serving.

    Regression: status_provider() (and any other handler) raising escaped the
    per-update loop and tore down run_command_bot entirely, taking the whole
    command bot offline until a restart.
    """
    stop_event = threading.Event()
    sent: list[str] = []
    batches = [
        [{"update_id": 1, "message": {"text": "/status", "from": {"id": 111}, "chat": {"id": 123}}}],
        [{"update_id": 2, "message": {"text": "/userinfo", "from": {"id": 111}, "chat": {"id": 123}}}],
    ]
    calls = {"n": 0}

    def get(_url, params, timeout):
        index = calls["n"]
        calls["n"] += 1
        return Response({"result": batches[index] if index < len(batches) else []})

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent.append(json["text"])
        if "user ID" in json["text"]:
            stop_event.set()
        return Response()

    def status_provider() -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        status_provider=status_provider,
        stop_event=stop_event,
        poll_timeout_seconds=0,
    )

    # The /status handler raised, but the bot kept polling and answered /userinfo.
    assert any("user ID" in message for message in sent)


def test_captioned_photo_command_keeps_its_argument(monkeypatch) -> None:
    """A photo captioned '/find percy' must pass 'percy' to the find provider.

    Regression: argument extraction read message['text'], which is empty for a
    photo (its text rides in 'caption'), so a captioned command lost its
    argument and ran as a bare '/find'.
    """
    stop_event = threading.Event()
    seen_targets: list[str] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "photo": [{"file_id": "small"}, {"file_id": "BIG"}],
                            "caption": "/find percy",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        return Response()

    def find_provider(chat_id: int, target: str) -> str:
        seen_targets.append(target)
        stop_event.set()
        return "On it."

    monkeypatch.setattr(
        "lib.telegram.commands.download_telegram_file", lambda base, fid, timeout=30: b"jpeg"
    )
    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        find_provider=find_provider,
        photo_provider=lambda image: "📸 ok",
    )

    assert seen_targets == ["percy"]


def test_free_text_with_ai_off_tells_allowed_user_instead_of_silence(monkeypatch) -> None:
    """Free text from an allowed user with nl_provider=None gets an "AI off" reply.

    Regression: with OLLAMA_ENABLED=0 the bot silently dropped non-command
    messages, which reads as a broken bot; it must say the AI chat is off.
    """
    stop_event = threading.Event()
    sent_messages: list[str] = []

    def post(_url, json, timeout):
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    monkeypatch.setattr(
        "lib.telegram.commands.requests.get", _single_update("how are the birds?")
    )
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
    )

    assert sent_messages == ["🤖 AI chat is off (OLLAMA_ENABLED=0) — use a /command."]


def test_free_text_from_unauthorized_user_gets_no_reply(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    polls = {"n": 0}

    def get(_url, params, timeout):
        # Serve the stranger's message once, then stop after a second poll so
        # the loop has fully handled the update before we assert on silence.
        polls["n"] += 1
        if polls["n"] == 1:
            return Response(
                {
                    "result": [
                        {
                            "update_id": 1,
                            "message": {
                                "text": "hello there",
                                "from": {"id": 999},
                                "chat": {"id": 123},
                            },
                        }
                    ]
                }
            )
        stop_event.set()
        return Response({"result": []})

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
    )

    # A stranger's free text is ignored entirely — no "AI off" hint, no
    # "Unauthorized." — so the bot never reveals itself to unknown senders.
    assert sent_messages == []


def test_find_command_registers_ack_message_for_progress_edits(monkeypatch) -> None:
    stop_event = threading.Event()
    attached: list[int | None] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/find percy",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(url, json, timeout):
        if "text" in json:
            stop_event.set()
            return Response({"result": {"message_id": 88}})
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        find_provider=lambda _chat_id, _target: "On it.",
        find_progress_message=attached.append,
    )

    assert attached == [88]


def test_tokens_command_replies_with_usage_report(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/tokens",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        stop_event.set()
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        tokens_provider=lambda: "🔤 LLM usage since Jul 04",
    )

    assert sent_messages == ["🔤 LLM usage since Jul 04"]


def test_memory_command_acks_then_edits_in_report(monkeypatch) -> None:
    # /memory scans disk off the poll loop: an immediate ack message, then the
    # report edited in when the scan finishes.
    stop_event = threading.Event()
    calls: list[tuple[str, str]] = []  # (api_method, text)
    done = threading.Event()

    def get(_url, params, timeout):
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/memory",
                            "from": {"id": 111},
                            "chat": {"id": 123},
                        },
                    }
                ]
            }
        )

    def post(url, json, timeout):
        if "text" not in json:
            return Response()
        method = url.rsplit("/", 1)[-1]
        calls.append((method, json["text"]))
        stop_event.set()
        if method == "editMessageText":
            done.set()
        return Response({"result": {"message_id": 55}})

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        memory_stats_provider=lambda: "🧠 Memory\nDays: 9",
    )
    assert done.wait(5.0), "report edit never happened"

    assert calls[0][0] == "sendMessage" and "Crunching" in calls[0][1]
    assert calls[-1][0] == "editMessageText" and "Days: 9" in calls[-1][1]


def test_memory_and_tokens_commands_require_allowed_user(monkeypatch) -> None:
    stop_event = threading.Event()
    sent_messages: list[str] = []
    seen = []

    def get(_url, params, timeout):
        if seen:
            stop_event.set()
            return Response({"result": []})
        seen.append(True)
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {"text": "/memory", "from": {"id": 999}, "chat": {"id": 123}},
                    },
                    {
                        "update_id": 2,
                        "message": {"text": "/tokens", "from": {"id": 999}, "chat": {"id": 123}},
                    },
                ]
            }
        )

    def post(_url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        return Response()

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    run_command_bot(
        "token",
        allowed_user_ids=["111"],
        stop_event=stop_event,
        poll_timeout_seconds=0,
        memory_stats_provider=lambda: "stats",
        tokens_provider=lambda: "tokens",
    )

    assert sent_messages == ["Unauthorized.", "Unauthorized."]


def test_memory_command_scans_one_at_a_time(monkeypatch) -> None:
    # A batch of queued /memory taps must not launch parallel full-disk scans:
    # the second tap gets a "hang on" while the first scan is still running.
    stop_event = threading.Event()
    sent_messages: list[str] = []
    release_scan = threading.Event()
    done = threading.Event()
    seen = []

    def get(_url, params, timeout):
        if seen:
            stop_event.set()
            return Response({"result": []})
        seen.append(True)
        return Response(
            {
                "result": [
                    {
                        "update_id": 1,
                        "message": {"text": "/memory", "from": {"id": 111}, "chat": {"id": 123}},
                    },
                    {
                        "update_id": 2,
                        "message": {"text": "/memory", "from": {"id": 111}, "chat": {"id": 123}},
                    },
                ]
            }
        )

    def post(url, json, timeout):
        if "text" not in json:
            return Response()
        sent_messages.append(json["text"])
        if url.endswith("/editMessageText"):
            done.set()
        return Response({"result": {"message_id": 9}})

    def slow_provider() -> str:
        release_scan.wait(5.0)
        return "🧠 Memory report"

    monkeypatch.setattr("lib.telegram.commands.requests.get", get)
    monkeypatch.setattr("lib.telegram.commands.requests.post", post)

    bot = threading.Thread(
        target=lambda: run_command_bot(
            "token",
            allowed_user_ids=["111"],
            stop_event=stop_event,
            poll_timeout_seconds=0,
            memory_stats_provider=slow_provider,
        ),
        daemon=True,
    )
    bot.start()
    # Both taps handled: one scan started, the other turned away immediately.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if any("hang on" in m for m in sent_messages):
            break
        time.sleep(0.02)
    release_scan.set()
    assert done.wait(5.0), "the running scan never delivered its report"
    bot.join(timeout=5.0)

    assert sum(1 for m in sent_messages if "Crunching" in m) == 1
    assert sum(1 for m in sent_messages if "hang on" in m) == 1
    assert any("Memory report" in m for m in sent_messages)
