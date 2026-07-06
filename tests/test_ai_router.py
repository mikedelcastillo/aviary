from __future__ import annotations

import time

import pytest

from lib.ai.client import OllamaUnavailableError
from lib.ai.intent import Intent
from lib.ai.router import QUEUED_ACK, UNREACHABLE_NOTICE, NaturalLanguageRouter
from lib.clock import now_ph
from lib.pending import PendingMessageStore


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


class RecordingClient:
    """Returns a fixed intent but remembers every messages list it was sent."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.seen: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.seen.append(messages)
        return self.content


def test_followup_carries_previous_turn_as_context() -> None:
    client = RecordingClient('{"action": "activity", "argument": "draft"}')
    dispatched: list = []
    router = _router(client, dispatched, [])

    # First question establishes the prior turn (an activity question).
    router._handle(5, "did draft or pizza fly today?")
    # Follow-up: the second classify call must see the first message as context.
    router._handle(5, "what about jynx or matcha?")

    assert len(client.seen) == 2
    followup_user = client.seen[1][-1]["content"]
    assert "did draft or pizza fly today?" in followup_user
    assert "what about jynx or matcha?" in followup_user


def test_first_turn_has_no_prior_context() -> None:
    client = RecordingClient('{"action": "find", "argument": "percy"}')
    router = _router(client, [], [])
    router._handle(9, "where is percy?")
    # Nothing before it -> the message is sent verbatim, no context preamble.
    assert client.seen[0][-1]["content"] == "where is percy?"


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


class UnavailableClient:
    """Simulates the health-gated 'backend known down' failure mode."""

    def chat(self, model, messages, **kwargs):
        raise OllamaUnavailableError("all olla nodes down")


class ExplodingClient:
    """Raises a plain transport error, as requests does mid-outage."""

    def chat(self, model, messages, **kwargs):
        raise ConnectionError("connection refused")


def test_unavailable_llm_queues_message_and_acks(tmp_path) -> None:
    # Outage + a pending store: the message must be SAVED (not dropped with an
    # apology), the user told once via QUEUED_ACK, and the buffer cleared so the
    # same text isn't re-prepended to the next turn.
    store = PendingMessageStore(tmp_path)
    dispatched: list = []
    notified: list = []
    router = NaturalLanguageRouter(
        UnavailableClient(),
        "qwen3:4b",
        lambda: ["percy"],
        dispatch=lambda chat_id, intent, text: dispatched.append((chat_id, intent, text)),
        notify=lambda chat_id, text: notified.append((chat_id, text)),
        pending=store,
    )

    router._handle(42, "where is percy?")

    assert dispatched == []
    assert [(item.chat_id, item.text) for item in store.items()] == [(42, "where is percy?")]
    assert notified == [(42, QUEUED_ACK)]
    assert router._buffers[42] == []  # turn committed even though it queued


def test_second_queued_message_is_saved_but_not_reacked(tmp_path) -> None:
    # A burst of questions during one outage earns ONE ack, not one per message;
    # later texts queue silently behind the first.
    store = PendingMessageStore(tmp_path)
    notified: list = []
    router = NaturalLanguageRouter(
        UnavailableClient(),
        "qwen3:4b",
        lambda: [],
        dispatch=lambda chat_id, intent, text: None,
        notify=lambda chat_id, text: notified.append((chat_id, text)),
        pending=store,
    )

    router._handle(42, "where is percy?")
    router._handle(42, "and how did they sleep?")

    assert store.count() == 2  # both saved for replay
    assert sum(1 for _, text in notified if text == QUEUED_ACK) == 1  # ack throttled


def test_unavailable_llm_without_store_keeps_old_apology() -> None:
    # No pending store wired -> the pre-queue behavior: apologize, drop, no dispatch.
    dispatched: list = []
    notified: list = []
    router = _router(UnavailableClient(), dispatched, notified)

    router._handle(42, "where is percy?")

    assert dispatched == []
    assert notified == [(42, UNREACHABLE_NOTICE)]


def test_replay_pending_dispatches_safe_intent_after_catchup_notice() -> None:
    # A read-only question queued during an outage is answered on recovery, with
    # a "catching up" line quoting the original FIRST so the answer isn't
    # mistaken for a reply to something current.
    events: list = []
    router = NaturalLanguageRouter(
        FakeClient('{"action": "status", "argument": ""}'),
        "qwen3:4b",
        lambda: [],
        dispatch=lambda chat_id, intent, text: events.append(("dispatch", chat_id, intent, text)),
        notify=lambda chat_id, text: events.append(("notify", chat_id, text)),
    )

    router.replay_pending(42, "how are things", now_ph())

    assert [kind for kind, *_ in events] == ["notify", "dispatch"]
    notice = events[0][2]
    assert "Catching up" in notice
    assert "how are things" in notice  # the original text is quoted back
    assert events[1] == ("dispatch", 42, Intent("status", ""), "how are things")


def test_replay_pending_refuses_stale_state_changing_command() -> None:
    # "pause" queued hours ago must NOT fire late; the user is told what was
    # skipped (with their text quoted) so they can resend it if still wanted.
    dispatched: list = []
    notified: list = []
    router = _router(
        FakeClient('{"action": "pause", "argument": "10m"}'), dispatched, notified
    )

    router.replay_pending(42, "stop the cams for ten minutes", now_ph())

    assert dispatched == []
    assert len(notified) == 1
    notice = notified[0][1]
    assert "won't run it late" in notice
    assert '"stop the cams for ten minutes"' in notice


def test_replay_pending_propagates_transport_errors() -> None:
    # The backfill worker retries on transport failure, so replay_pending must
    # let the exception escape rather than consuming the message.
    router = _router(ExplodingClient(), [], [])

    with pytest.raises(ConnectionError):
        router.replay_pending(42, "how are things", now_ph())


def test_replay_pending_reports_dispatch_error_without_raising() -> None:
    # A handler blowing up during replay is surfaced to the user and consumed —
    # it must not propagate (the worker would retry a poisoned message forever).
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

    router.replay_pending(42, "how are things", now_ph())

    assert notified[-1][1] == "Something went wrong handling that — sorry!"


class UnavailableClient:
    def chat(self, model, messages, **kwargs):
        raise OllamaUnavailableError("gated")


def test_stale_generation_failure_queues_nothing(tmp_path) -> None:
    # A newer bubble arrived while the OLD generation's classify was failing:
    # the stale turn must queue nothing (its text is still in the buffer for the
    # newer generation), or the user would be answered twice on replay.
    store = PendingMessageStore(tmp_path / "pending")
    notified: list = []
    router = NaturalLanguageRouter(
        UnavailableClient(), "gemma3:4b", lambda: [],
        dispatch=lambda *a: None,
        notify=lambda chat_id, text: notified.append((chat_id, text)),
        pending=store,
    )
    with router._lock:
        router._buffers[5] = ["where is percy", "and pizza?"]
        router._generation[5] = 2  # a newer message superseded generation 1
    router._process(5, 1)  # the stale generation fails its classify
    assert store.count() == 0 and notified == []
    router._process(5, 2)  # the current generation queues the COMBINED text
    assert [i.text for i in store.items()] == ["where is percy and pizza?"]
    assert notified == [(5, QUEUED_ACK)]


def test_queue_ack_repeats_after_window_expiry(tmp_path, monkeypatch) -> None:
    # The once-per-window ack throttle must re-ack after QUEUE_ACK_SECONDS — a
    # user queueing again much later deserves fresh confirmation.
    from types import SimpleNamespace
    import lib.ai.router as router_module
    now = [1000.0]
    monkeypatch.setattr(
        router_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    store = PendingMessageStore(tmp_path / "pending")
    notified: list = []
    router = NaturalLanguageRouter(
        UnavailableClient(), "gemma3:4b", lambda: [],
        dispatch=lambda *a: None,
        notify=lambda chat_id, text: notified.append(text),
        pending=store,
    )
    router.queue_message(9, "first")
    now[0] += 100.0
    router.queue_message(9, "second")  # inside the window: silent
    now[0] += router_module.QUEUE_ACK_SECONDS + 1.0
    router.queue_message(9, "third")  # window expired: ack again
    assert notified == [QUEUED_ACK, QUEUED_ACK]
    assert store.count() == 3


def test_queue_save_failure_falls_back_to_apology(tmp_path) -> None:
    # If persisting the message itself fails, the user must still hear SOMETHING
    # — the plain unreachable apology, never silence.
    class BrokenStore:
        def add(self, chat_id, text, when):
            raise OSError("disk full")

    notified: list = []
    router = NaturalLanguageRouter(
        UnavailableClient(), "gemma3:4b", lambda: [],
        dispatch=lambda *a: None,
        notify=lambda chat_id, text: notified.append(text),
        pending=BrokenStore(),
    )
    router.queue_message(9, "hello?")
    assert notified == [UNREACHABLE_NOTICE]


def test_replay_dispatch_outage_propagates_and_clears_replay_flag() -> None:
    # The cluster dying mid-replay (classify OK, dispatch's chat call gated)
    # must PROPAGATE so the worker keeps the same item and its attempt count —
    # and the thread's replay flag must be cleared even then.
    notified: list = []

    def dispatch(chat_id, intent, text):
        raise OllamaUnavailableError("died mid-replay")

    router = NaturalLanguageRouter(
        FakeClient('{"action": "chat", "argument": ""}'), "gemma3:4b", lambda: [],
        dispatch=dispatch,
        notify=lambda chat_id, text: notified.append(text),
    )
    with pytest.raises(OllamaUnavailableError):
        router.replay_pending(3, "how are the birds?", now_ph())
    assert router.replaying() is False  # flag cleared on the failure path
    # Only the catching-up notice went out; no misleading "went wrong" line for
    # an outage the worker is about to retry.
    assert len(notified) == 1 and "Catching up" in notified[0]


def test_dispatch_sees_replaying_flag_during_replay() -> None:
    # Downstream outage handlers (send_chat_reply, the activity hook) key off
    # replaying() to re-raise instead of re-queueing; it must be True exactly
    # while a replayed dispatch runs on this thread.
    seen: list = []
    router = NaturalLanguageRouter(
        FakeClient('{"action": "chat", "argument": ""}'), "gemma3:4b", lambda: [],
        dispatch=lambda chat_id, intent, text: seen.append(None),
        notify=lambda chat_id, text: None,
    )
    router._dispatch = lambda chat_id, intent, text: seen.append(router.replaying())
    router.replay_pending(3, "hello", now_ph())
    assert seen == [True]
    assert router.replaying() is False
