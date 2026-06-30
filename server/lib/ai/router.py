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
from typing import Callable

from lib.ai.client import OllamaClient
from lib.ai.intent import classify_intent


LOGGER = logging.getLogger("lib.ai.router")

# Quiet gap after the last message before we process (coalesces multi-bubble).
DEFAULT_DEBOUNCE_SECONDS = 1.3
# Re-assert the typing indicator this often during a long think (Telegram clears
# it after ~5s).
TYPING_PULSE_SECONDS = 4.0


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
    ) -> None:
        self._client = client
        self._model = model
        self._findable_birds = findable_birds
        self._dispatch = dispatch
        self._notify = notify
        self._typing = typing
        self._debounce = debounce_seconds
        self._lock = threading.Lock()
        self._buffers: dict[int, list[str]] = {}
        self._generation: dict[int, int] = {}
        self._timers: dict[int, threading.Timer] = {}

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
                    self._client, self._model, combined, self._findable_birds()
                )
            except Exception:
                LOGGER.exception("Intent classification failed")
                self._notify(
                    chat_id,
                    "🤖 My language brain (Ollama) is unreachable right now — try a /command.",
                )
                self._commit(chat_id, generation)
                return

            # A newer message arrived while we were classifying — discard this
            # result; the newer (fuller) prompt will be processed instead.
            if not self._is_current(chat_id, generation):
                LOGGER.info("Discarding stale NL response for chat %s", chat_id)
                return

            self._commit(chat_id, generation)
            LOGGER.info("NL %r -> action=%s arg=%r", combined[:60], intent.action, intent.argument)
            try:
                self._dispatch(chat_id, intent, combined)
            except Exception:
                LOGGER.exception("NL dispatch failed for action=%s", intent.action)
                self._notify(chat_id, "Something went wrong handling that — sorry!")
        finally:
            stop_typing.set()

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
