"""Answer activity questions from the persistent day memory (the journal).

``/activity``, "what did percy do today?", "what did percy and pizza do lately?"
all read the Markdown memory the caretaker keeps (:mod:`lib.journal`) — which
already holds VLM-written notes and the photos it looked at — and fold the
relevant entries into a <=3-sentence report, with a few of the best photos. If a
specific bird hasn't been logged recently and the question is about *now*, it
kicks off a live ``/find`` instead of shrugging.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from lib.activity import summarise_activity
from lib.find import pretty_phrase
from lib.journal import load_recent
from lib.roster import expand_targets


LOGGER = logging.getLogger("lib.activity_qa")

MAX_QA_PHOTOS = 4
SUMMARY_TIMEOUT_SECONDS = 60.0

# Words that make an empty result trigger a live search instead of "haven't seen".
_LIVE_WORDS = ("now", "right now", "currently", "doing", "up to", "where")


def parse_activity_arg(argument: str) -> tuple[str, bool]:
    """Split a /activity argument into (bird text, today?). "percy today" -> ("percy", True)."""
    tokens = argument.strip().split()
    today = any(t.lower() in ("today", "day") for t in tokens)
    birds = " ".join(t for t in tokens if t.lower() not in ("today", "day"))
    return birds.strip(), today


class ActivityResponder:
    def __init__(
        self,
        memories_dir,
        client,
        llm_model: str,
        known_labels: Callable[[], list[str]],
        *,
        notify: Callable[[int, str], None],
        send_photo: Callable[[int, bytes, str | None], object] | None = None,
        find: Callable[[int, str], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._memories_dir = Path(memories_dir)
        self._client = client
        self._llm_model = llm_model
        self._known_labels = known_labels
        self._notify = notify
        self._send_photo = send_photo
        self._find = find
        self._clock = clock

    def _window(self, text: str, argument: str, now: datetime) -> tuple[datetime, str]:
        if "today" in text.lower() or "today" in argument.lower():
            return now.replace(hour=0, minute=0, second=0, microsecond=0), "today"
        return now - timedelta(hours=1), "in the last hour"

    def respond(self, chat_id: int, text: str, argument: str) -> None:
        bird_text, _ = parse_activity_arg(argument)
        now = datetime.fromtimestamp(self._clock())
        since, window_phrase = self._window(text, argument, now)
        targets = expand_targets(bird_text, self._known_labels()) if bird_text.strip() else None

        entries = load_recent(
            self._memories_dir, since, now, set(targets) if targets else None
        )
        if not entries:
            who = pretty_phrase(bird_text) if bird_text.strip() else "the birds"
            wants_live = any(word in text.lower() for word in _LIVE_WORDS)
            if targets and self._find is not None and wants_live:
                self._notify(chat_id, f"I haven't logged {who} {window_phrase} — let me check the cameras…")
                self._find(chat_id, bird_text)
                return
            self._notify(chat_id, f"I haven't logged any activity for {who} {window_phrase}.")
            return

        notes = [f"{entry.time.strftime('%H:%M')} — {entry.note}" for entry in entries]
        subject = pretty_phrase(bird_text) if bird_text.strip() else ""
        try:
            summary = summarise_activity(
                self._client, self._llm_model, notes, subject, timeout_seconds=SUMMARY_TIMEOUT_SECONDS
            )
        except Exception:
            LOGGER.exception("Activity summary failed")
            summary = notes[-1]
        self._notify(chat_id, summary or notes[-1])

        # A few of the most recent distinct photos (newest entries first).
        if self._send_photo is not None:
            chosen: list[str] = []
            seen: set[str] = set()
            for entry in reversed(entries):
                for photo in entry.photos:
                    if photo not in seen and Path(photo).exists():
                        seen.add(photo)
                        chosen.append(photo)
                    if len(chosen) >= MAX_QA_PHOTOS:
                        break
                if len(chosen) >= MAX_QA_PHOTOS:
                    break
            for photo in reversed(chosen):
                try:
                    self._send_photo(chat_id, Path(photo).read_bytes(), None)
                except Exception:
                    LOGGER.exception("Sending activity photo failed")
