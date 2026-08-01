"""Route free-text messages: coalesce rapid bubbles, classify, then dispatch.

People type in several quick bubbles. So instead of answering each one, the
router DEBOUNCES per chat: each new message resets a short timer and bumps a
generation counter; only after a quiet gap does it process the COMBINED text.
A response already in flight when a newer message lands is discarded (its
generation is stale) and the fuller prompt is processed instead. While it works,
it shows a "typing…" indicator so the user knows it's thinking.

Intent classification is delegated to :mod:`lib.ai.intent`; the actual
command/chat dispatch is a callback supplied by ``main``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable

from lib.ai.client import OllamaClient, OllamaUnavailableError
from lib.ai.intent import classify_intent
from lib.clock import now_ph


LOGGER = logging.getLogger("lib.ai.router")

# Quiet gap after the last message before we process (coalesces multi-bubble).
DEFAULT_DEBOUNCE_SECONDS = 1.3
# Re-assert the typing indicator this often during a long think (Telegram clears
# it after ~5s).
TYPING_PULSE_SECONDS = 4.0
# How long a previous turn stays eligible as follow-up context. After this, an
# unrelated later question won't be mis-read as continuing an old one.
FOLLOWUP_CONTEXT_SECONDS = 300.0

# During an outage, tell each chat its message was SAVED (not dropped) — but
# only once per this window; later messages queue silently behind the first ack
# so a burst of questions doesn't earn a burst of identical apologies.
QUEUE_ACK_SECONDS = 600.0
QUEUED_ACK = (
    "💤 My language brain (Ollama) is offline right now — I saved your message "
    "and will answer as soon as it's back."
)
UNREACHABLE_NOTICE = "🤖 My language brain (Ollama) is unreachable right now — try a /command."

# Intents safe to EXECUTE late when a queued message is replayed after recovery.
# Read-only questions age gracefully; state-changing commands (pause, restart,
# a PTZ patrol…) must not fire minutes or hours after they were asked for.
REPLAY_SAFE_ACTIONS = frozenset(
    {"chat", "activity", "sleep", "weather", "status", "machine", "snapshot"}
)
# Keep the quoted original short in replay prefixes.
REPLAY_QUOTE_CHARS = 200


class NaturalLanguageRouter:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        findable_birds: Callable[[], list[str]],
        dispatch: Callable[[int, "object", str], None],
        notify: Callable[[int, str], None],
        *,
        typing: Callable[[int], None] | None = None,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        pending=None,
        on_supersede: Callable[[int], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._findable_birds = findable_birds
        self._dispatch = dispatch
        self._notify = notify
        self._typing = typing
        self._debounce = debounce_seconds
        # Optional lib.pending.PendingMessageStore: messages that fail on an
        # unreachable LLM are SAVED here (and replayed after recovery by the
        # backfill worker) instead of dropped with an apology.
        self._pending = pending
        # Optional hook fired on EVERY incoming message: anything still
        # streaming a reply to this chat is stale the moment the user says
        # something new, so main wires this to the stream registry's cancel.
        self._on_supersede = on_supersede
        self._lock = threading.Lock()
        self._buffers: dict[int, list[str]] = {}
        self._generation: dict[int, int] = {}
        self._timers: dict[int, threading.Timer] = {}
        self._last_queue_ack: dict[int, float] = {}
        # True on a thread currently dispatching a REPLAYED message. Downstream
        # outage handlers check this: during a replay they must re-raise (so the
        # backfill worker keeps the SAME item, preserving its attempt count)
        # rather than queue a fresh copy — self-requeueing reset the give-up
        # counter and could loop forever across cluster flaps.
        self._replay_local = threading.local()
        # Previous committed turn per chat: (text, action, monotonic_time). Lets a
        # short follow-up ("what about percy?") inherit the prior turn's action
        # instead of defaulting to `find`.
        self._last_turn: dict[int, tuple[str, str, float]] = {}

    def handle_async(self, chat_id: int, text: str) -> None:
        """Queue a message and (re)arm the debounce timer for its chat."""
        with self._lock:
            self._buffers.setdefault(chat_id, []).append(text)
            self._generation[chat_id] = self._generation.get(chat_id, 0) + 1
            generation = self._generation[chat_id]
            timer = self._timers.pop(chat_id, None)
            if timer is not None:
                timer.cancel()
            new_timer = threading.Timer(self._debounce, self._fire, args=(chat_id, generation))
            new_timer.daemon = True
            self._timers[chat_id] = new_timer
            new_timer.start()
        if self._on_supersede is not None:
            # Outside the lock: the hook touches the stream registry, never us.
            try:
                self._on_supersede(chat_id)
            except Exception:
                LOGGER.exception("Stream supersede hook failed")

    def _fire(self, chat_id: int, generation: int) -> None:
        threading.Thread(
            target=self._process, args=(chat_id, generation), name="nl-router", daemon=True
        ).start()

    def _is_current(self, chat_id: int, generation: int) -> bool:
        with self._lock:
            return self._generation.get(chat_id) == generation

    def _process(self, chat_id: int, generation: int) -> None:
        if not self._is_current(chat_id, generation):
            return  # superseded by a newer message before we even started
        with self._lock:
            combined = " ".join(self._buffers.get(chat_id, [])).strip()
        if not combined:
            return

        stop_typing = threading.Event()
        self._start_typing(chat_id, stop_typing)
        try:
            try:
                intent = classify_intent(
                    self._client,
                    self._model,
                    combined,
                    self._findable_birds(),
                    prior=self._prior_turn(chat_id),
                )
            except Exception as exc:
                # Mirror the success path's staleness check: if a newer bubble
                # superseded this generation while classify was failing, its text
                # is still in the buffer — the newer generation will queue the
                # fuller combined prompt itself. Queueing here too would save the
                # partial AND the superset, and answer the user twice on replay.
                if not self._is_current(chat_id, generation):
                    LOGGER.info("Discarding stale failed NL turn for chat %s", chat_id)
                    return
                if isinstance(exc, OllamaUnavailableError):
                    LOGGER.warning("LLM unavailable for chat %s; queueing message", chat_id)
                else:
                    LOGGER.exception("Intent classification failed; queueing message")
                self.queue_message(chat_id, combined)
                self._commit(chat_id, generation)
                return

            # A newer message arrived while we were classifying — discard this
            # result; the newer (fuller) prompt will be processed instead.
            if not self._is_current(chat_id, generation):
                LOGGER.info("Discarding stale NL response for chat %s", chat_id)
                return

            self._commit(chat_id, generation)
            self._record_turn(chat_id, combined, intent.action)
            LOGGER.info("NL %r -> action=%s arg=%r", combined[:60], intent.action, intent.argument)
            try:
                self._dispatch(chat_id, intent, combined)
            except Exception:
                LOGGER.exception("NL dispatch failed for action=%s", intent.action)
                self._notify(chat_id, "Something went wrong handling that — sorry!")
        finally:
            stop_typing.set()

    def queue_message(self, chat_id: int, text: str) -> None:
        """Save a message the LLM couldn't process, and tell the user ONCE.

        Falls back to the plain "unreachable" apology when there is no store (or
        the save itself fails) — the user should never get silence.
        """
        if self._pending is None:
            self._notify(chat_id, UNREACHABLE_NOTICE)
            return
        try:
            self._pending.add(chat_id, text, now_ph())
        except Exception:
            LOGGER.exception("Saving pending message failed")
            self._notify(chat_id, UNREACHABLE_NOTICE)
            return
        now = time.monotonic()
        with self._lock:
            last = self._last_queue_ack.get(chat_id)
            acked_recently = last is not None and (now - last) < QUEUE_ACK_SECONDS
            self._last_queue_ack[chat_id] = now
        if not acked_recently:
            self._notify(chat_id, QUEUED_ACK)

    def replay_pending(self, chat_id: int, text: str, created: datetime) -> None:
        """Answer one queued message now that the LLM is back.

        Read-only intents are dispatched normally (with a "catching up" line so
        the answer isn't mistaken for a reply to something current); an intent
        that would CHANGE state late is surfaced back to the user instead of
        executed. Raises on a transport error so the backfill worker keeps the
        message and retries later; a dispatch failure is reported and consumed.
        """
        intent = classify_intent(
            self._client, self._model, text, self._findable_birds(),
            prior=self._prior_turn(chat_id),
        )
        quoted = text if len(text) <= REPLAY_QUOTE_CHARS else text[: REPLAY_QUOTE_CHARS - 1] + "…"
        stamp = created.strftime("%H:%M") if created.date() == now_ph().date() else created.strftime("%a %H:%M")
        if intent.action not in REPLAY_SAFE_ACTIONS:
            LOGGER.info("Not replaying stale %s command from chat %s", intent.action, chat_id)
            self._notify(
                chat_id,
                f'↩️ While I was offline you asked ({stamp}): "{quoted}" — that looks like a '
                "command, so I won't run it late. Send it again if you still want it.",
            )
            return
        LOGGER.info("Replaying queued NL %r -> action=%s", text[:60], intent.action)
        self._record_turn(chat_id, text, intent.action)
        self._notify(chat_id, f'↩️ Catching up on your message from {stamp}: "{quoted}"')
        self._replay_local.active = True
        try:
            self._dispatch(chat_id, intent, text)
        except OllamaUnavailableError:
            # The cluster died again mid-replay — propagate so the worker keeps
            # this item (and its attempt count) instead of consuming it.
            raise
        except Exception:
            LOGGER.exception("Replay dispatch failed for action=%s", intent.action)
            self._notify(chat_id, "Something went wrong handling that — sorry!")
        finally:
            self._replay_local.active = False

    def replaying(self) -> bool:
        """Whether THIS thread is inside a replayed dispatch (see _replay_local)."""
        return bool(getattr(self._replay_local, "active", False))

    def _prior_turn(self, chat_id: int) -> tuple[str, str] | None:
        """The previous turn's (text, action) if it's recent enough to be a
        follow-up context; ``None`` otherwise."""
        with self._lock:
            prev = self._last_turn.get(chat_id)
        if prev is None:
            return None
        text, action, when = prev
        if time.monotonic() - when > FOLLOWUP_CONTEXT_SECONDS:
            return None
        return text, action

    def _record_turn(self, chat_id: int, text: str, action: str) -> None:
        with self._lock:
            self._last_turn[chat_id] = (text, action, time.monotonic())

    def _commit(self, chat_id: int, generation: int) -> None:
        """Clear the buffer once this generation owns the conversation turn."""
        with self._lock:
            if self._generation.get(chat_id) == generation:
                self._buffers[chat_id] = []

    def _start_typing(self, chat_id: int, stop: threading.Event) -> None:
        if self._typing is None:
            return

        def pulse() -> None:
            try:
                self._typing(chat_id)
            except Exception:
                pass
            while not stop.wait(TYPING_PULSE_SECONDS):
                try:
                    self._typing(chat_id)
                except Exception:
                    pass

        threading.Thread(target=pulse, name="nl-typing", daemon=True).start()

    # Kept for tests / callers that want the old synchronous path.
    def _handle(self, chat_id: int, text: str) -> None:
        with self._lock:
            self._buffers[chat_id] = [text]
            self._generation[chat_id] = self._generation.get(chat_id, 0) + 1
            generation = self._generation[chat_id]
        self._process(chat_id, generation)

    def _clock(self) -> float:  # pragma: no cover - retained for compatibility
        return time.monotonic()
