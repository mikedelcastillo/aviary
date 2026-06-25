from __future__ import annotations

import time

from lib.ai.intent import Intent
from lib.ai.router import NaturalLanguageRouter


class FakeClient:
    def __init__(self, content: str = "", *, boom: bool = False) -> None:
        self.content = content
        self.boom = boom

    def chat(self, model, messages, **kwargs):
        if self.boom:
            raise RuntimeError("ollama down")
        return self.content


def _router(client, dispatched, notified):
    return NaturalLanguageRouter(
        client,
        "qwen3:4b",
        lambda: ["percy", "matcha"],
        dispatch=lambda chat_id, intent, text: dispatched.append((chat_id, intent, text)),
        notify=lambda chat_id, text: notified.append((chat_id, text)),
    )


def test_handle_classifies_then_dispatches() -> None:
    dispatched: list = []
    notified: list = []
    router = _router(FakeClient('{"action": "pause", "argument": "10m"}'), dispatched, notified)

    router._handle(123, "stop the cams for ten minutes")

    assert dispatched == [(123, Intent("pause", "10m"), "stop the cams for ten minutes")]
    assert notified == []  # dispatch owns the replies on the happy path


def test_handle_reports_when_model_unreachable() -> None:
    dispatched: list = []
    notified: list = []
    router = _router(FakeClient(boom=True), dispatched, notified)

    router._handle(123, "where is percy?")

    assert dispatched == []
    assert len(notified) == 1
    assert "unreachable" in notified[0][1].lower()


def test_handle_recovers_from_dispatch_error() -> None:
    notified: list = []

    def boom_dispatch(chat_id, intent, text):
        raise RuntimeError("handler blew up")

    router = NaturalLanguageRouter(
        FakeClient('{"action": "status", "argument": ""}'),
        "qwen3:4b",
        lambda: [],
        dispatch=boom_dispatch,
        notify=lambda chat_id, text: notified.append((chat_id, text)),
    )

    router._handle(123, "how are things")

    # A dispatch exception is caught and surfaced, never crashes the thread.
    assert len(notified) == 1
    assert "wrong" in notified[0][1].lower()


def test_coalesces_rapid_messages_into_one_dispatch() -> None:
    dispatched: list = []
    router = NaturalLanguageRouter(
        FakeClient('{"action": "find", "argument": "percy"}'),
        "qwen3:4b",
        lambda: ["percy"],
        dispatch=lambda chat_id, intent, text: dispatched.append(text),
        notify=lambda chat_id, text: None,
        debounce_seconds=0.05,
    )
    # Three quick bubbles -> debounced into a single combined dispatch.
    router.handle_async(1, "where")
    router.handle_async(1, "is")
    router.handle_async(1, "percy")
    time.sleep(0.3)
    assert dispatched == ["where is percy"]


def test_typing_indicator_is_pulsed() -> None:
    typed: list = []
    router = NaturalLanguageRouter(
        FakeClient('{"action": "status", "argument": ""}'),
        "qwen3:4b",
        lambda: [],
        dispatch=lambda chat_id, intent, text: None,
        notify=lambda chat_id, text: None,
        typing=lambda chat_id: typed.append(chat_id),
    )
    router._handle(7, "how are things")
    assert 7 in typed  # typing was signalled while thinking
