"""Short per-chat conversation memory so the bot keeps context across messages.

Holds the last ~20 turns per chat in memory (not persisted — a fresh start each
run is fine for casual chat). The chat reply path reads the history to stay
coherent ("and what about her?") and records each turn back.
"""

from __future__ import annotations

import threading
from collections import deque


class ConversationMemory:
    def __init__(self, max_messages: int = 20) -> None:
        self._max = max_messages
        self._lock = threading.Lock()
        self._chats: dict[int, deque] = {}

    def record(self, chat_id: int, role: str, content: str) -> None:
        if not content:
            return
        with self._lock:
            history = self._chats.get(chat_id)
            if history is None:
                history = deque(maxlen=self._max)
                self._chats[chat_id] = history
            history.append({"role": role, "content": content})

    def history(self, chat_id: int) -> list[dict[str, str]]:
        with self._lock:
            history = self._chats.get(chat_id)
            return list(history) if history else []
