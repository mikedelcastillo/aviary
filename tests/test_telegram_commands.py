from __future__ import annotations

import threading

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
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

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
        # Skip the startup setMyCommands registration; record only replies.
        if "text" not in json:
            return Response()
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
