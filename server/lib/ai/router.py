"""Route a free-text Telegram message: classify intent, then dispatch.

Each message runs on its own daemon thread so a slow model call never blocks the
Telegram poll loop. Intent classification is delegated to :mod:`lib.ai.intent`;
the actual command/chat dispatch is a callback supplied by ``main`` (where the
command providers live), keeping this module free of any command wiring.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from lib.ai.client import OllamaClient
from lib.ai.intent import Intent, classify_intent


LOGGER = logging.getLogger("lib.ai.router")


class NaturalLanguageRouter:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        findable_birds: Callable[[], list[str]],
        dispatch: Callable[[int, Intent, str], None],
        notify: Callable[[int, str], None],
    ) -> None:
        self._client = client
        self._model = model
        self._findable_birds = findable_birds
        self._dispatch = dispatch
        self._notify = notify

    def handle_async(self, chat_id: int, text: str) -> None:
        """Fire-and-forget: classify + dispatch ``text`` on a background thread."""
        threading.Thread(
            target=self._handle,
            args=(chat_id, text),
            name="nl-router",
            daemon=True,
        ).start()

    def _handle(self, chat_id: int, text: str) -> None:
        try:
            intent = classify_intent(self._client, self._model, text, self._findable_birds())
        except Exception:
            LOGGER.exception("Intent classification failed")
            self._notify(
                chat_id,
                "🤖 My language brain (Ollama) is unreachable right now — "
                "try a /command instead.",
            )
            return
        LOGGER.info("NL %r -> action=%s arg=%r", text[:60], intent.action, intent.argument)
        try:
            self._dispatch(chat_id, intent, text)
        except Exception:
            LOGGER.exception("NL dispatch failed for action=%s", intent.action)
            self._notify(chat_id, "Something went wrong handling that — sorry!")
