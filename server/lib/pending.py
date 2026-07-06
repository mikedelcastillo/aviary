"""Durable queue of messages that needed the LLM while it was unreachable.

When the Ollama cluster is down, a free-text Telegram message used to be
answered with an apology and DROPPED — the user had to notice and retype it.
Now the router saves it here instead; the backfill worker replays every saved
message (classify → dispatch → answer) the moment the cluster recovers, so
nothing typed during an outage is ever lost. The store survives restarts: a
message queued right before a crash is still answered on the next boot.

One JSON file per message (``<timestamp>_<seq>_<chat>.json``), written
atomically and deleted on completion — no compaction, no partial-line corruption,
and the queue is inspectable with ``ls``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path


LOGGER = logging.getLogger("lib.pending")


@dataclass(frozen=True)
class PendingMessage:
    id: str  # the filename stem; stable across restarts
    chat_id: int
    text: str
    created: datetime
    attempts: int = 0


class PendingMessageStore:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0

    def add(self, chat_id: int, text: str, when: datetime) -> PendingMessage:
        """Persist one message; the filename orders the queue chronologically."""
        with self._lock:
            self._seq += 1
            name = f"{when.strftime('%Y%m%d_%H%M%S_%f')}_{self._seq:03d}_{chat_id}"
        item = PendingMessage(id=name, chat_id=int(chat_id), text=text, created=when)
        self._write(item)
        LOGGER.info("Queued pending LLM message %s (chat %s)", name, chat_id)
        return item

    def items(self) -> list[PendingMessage]:
        """Every queued message, oldest first; unreadable files are skipped."""
        out: list[PendingMessage] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append(
                    PendingMessage(
                        id=path.stem,
                        chat_id=int(data["chat_id"]),
                        text=str(data["text"]),
                        created=datetime.fromisoformat(str(data["created"])),
                        attempts=int(data.get("attempts", 0)),
                    )
                )
            except (OSError, ValueError, KeyError):
                LOGGER.warning("Skipping unreadable pending message %s", path.name)
        return out

    def bump_attempts(self, item: PendingMessage) -> PendingMessage:
        """Record one more failed replay attempt; returns the updated item."""
        updated = replace(item, attempts=item.attempts + 1)
        self._write(updated)
        return updated

    def remove(self, item_id: str) -> None:
        try:
            (self._dir / f"{item_id}.json").unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Removing pending message %s failed", item_id)

    def count(self) -> int:
        return sum(1 for _ in self._dir.glob("*.json"))

    def _write(self, item: PendingMessage) -> None:
        # tmp + atomic rename: a crash mid-write can never leave a half-JSON
        # file that items() would then skip (losing the message it holds).
        path = self._dir / f"{item.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "chat_id": item.chat_id,
                    "text": item.text,
                    "created": item.created.isoformat(),
                    "attempts": item.attempts,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
