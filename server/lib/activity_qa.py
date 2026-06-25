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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from lib.activity import answer_activity_question, summarise_activity
from lib.clock import now_ph
from lib.find import pretty_phrase
from lib.journal import humanize_ago, load_recent
from lib.labels import pretty
from lib.roster import expand_targets


LOGGER = logging.getLogger("lib.activity_qa")

MAX_QA_PHOTOS = 4
SUMMARY_TIMEOUT_SECONDS = 60.0
CAPTION_LIMIT = 1024  # Telegram album/photo caption cap.

# Words that make an empty result trigger a live search instead of "haven't seen".
_LIVE_WORDS = ("now", "right now", "currently", "doing", "up to", "where")

# A message starting with one of these (or containing "?") is treated as a
# specific question to answer, not a request for a generic activity summary.
_QUESTION_WORDS = (
    "did", "do", "does", "is", "are", "was", "were", "when", "what", "where",
    "who", "how", "has", "have", "had", "can", "could", "will", "should", "any",
)


def _is_question(text: str) -> bool:
    stripped = text.strip().lstrip("/").lower()
    if "?" in stripped:
        return True
    first = stripped.split()[0] if stripped.split() else ""
    return first in _QUESTION_WORDS


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
        send_album: Callable[[int, list[tuple[bytes, str | None]]], object] | None = None,
        find: Callable[[int, str], None] | None = None,
        pronoun_note: str = "",
        now: Callable[[], datetime] = now_ph,
    ) -> None:
        self._memories_dir = Path(memories_dir)
        self._client = client
        self._llm_model = llm_model
        self._known_labels = known_labels
        self._notify = notify
        self._send_album = send_album
        self._find = find
        self._pronoun_note = pronoun_note
        self._now = now

    def _window(self, text: str, argument: str, now: datetime, question: bool) -> tuple[datetime, datetime, str]:
        """Resolve the lookback window ``(since, until, phrase)``, honouring
        time-of-day granularity.

        Explicit phrasing wins (morning/afternoon/evening/today/last hour).
        Otherwise a question defaults to the whole day ("did pizza eat?" means
        today), while a bare /activity defaults to the last hour. Windows are
        clamped so they never extend past ``now``.
        """
        t = f"{text} {argument}".lower()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        def clamp(end: datetime) -> datetime:
            return min(end, now)

        if "morning" in t:
            return midnight.replace(hour=5), clamp(midnight.replace(hour=12)), "this morning"
        if "afternoon" in t:
            return midnight.replace(hour=12), clamp(midnight.replace(hour=17)), "this afternoon"
        if "evening" in t or "tonight" in t or "night" in t:
            return midnight.replace(hour=17), now, "this evening"
        if "week" in t:
            return midnight - timedelta(days=6), now, "this week"
        if "yesterday" in t:
            y = midnight - timedelta(days=1)
            return y, midnight, "yesterday"
        if "today" in t or "all day" in t or "whole day" in t or "so far" in t:
            return midnight, now, "today"
        if "hour" in t or "recently" in t or "just now" in t or "lately" in t:
            return now - timedelta(hours=1), now, "in the last hour"
        # Questions and photo/together requests mean the whole day; only a bare
        # /activity summary defaults to the last hour.
        broad = question or any(
            w in t for w in ("photo", "picture", "show", "together", "with ", "spend")
        )
        if broad:
            return midnight, now, "today"
        return now - timedelta(hours=1), now, "in the last hour"

    def respond(self, chat_id: int, text: str, argument: str) -> None:
        bird_text, _ = parse_activity_arg(argument)
        now = self._now()
        question = _is_question(text)
        since, until, window_phrase = self._window(text, argument, now, question)
        targets = expand_targets(bird_text, self._known_labels()) if bird_text.strip() else None

        entries = load_recent(
            self._memories_dir, since, until, set(targets) if targets else None
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

        # Each note carries when it happened (relative, "2 hours ago") and which
        # birds were seen — so the model can answer "together?" / "when?" / "did X
        # do Y?" and weave timing into a summary.
        notes = [
            f"({humanize_ago(entry.time, now)}) "
            f"[{', '.join(pretty(b) for b in entry.birds) if entry.birds else 'quiet'}]: {entry.note}"
            for entry in entries
        ]
        try:
            if question:
                summary = answer_activity_question(
                    self._client, self._llm_model, text, notes,
                    self._pronoun_note, window_phrase, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                )
            else:
                subject = pretty_phrase(bird_text) if bird_text.strip() else ""
                summary = summarise_activity(
                    self._client, self._llm_model, notes, subject,
                    self._pronoun_note, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                )
        except Exception:
            LOGGER.exception("Activity response failed")
            summary = ""
        summary = summary or notes[-1]

        # Pick the photos that go with the answer. For a "together"/"with other
        # birds" request, prefer moments where two or more birds were seen at once
        # (the actual together-shots); otherwise the most recent relevant photos.
        wants_together = any(
            phrase in text.lower()
            for phrase in ("together", "with each other", "with other", "spend", "spending", "with ")
        )
        pool = entries
        if wants_together:
            multi = [e for e in entries if len(e.birds) >= 2]
            if multi:
                pool = multi
        chosen: list[str] = []
        seen: set[str] = set()
        for entry in reversed(pool):
            for photo in entry.photos:
                if photo not in seen and Path(photo).exists():
                    seen.add(photo)
                    chosen.append(photo)
                if len(chosen) >= MAX_QA_PHOTOS:
                    break
            if len(chosen) >= MAX_QA_PHOTOS:
                break

        # Send ONE album — photos grouped with the summary as the first caption —
        # rather than a text plus a burst of separate photos.
        if self._send_album is not None and chosen:
            items: list[tuple[bytes, str | None]] = []
            caption = summary if len(summary) <= CAPTION_LIMIT else summary[: CAPTION_LIMIT - 1] + "…"
            for index, photo in enumerate(reversed(chosen)):
                try:
                    items.append((Path(photo).read_bytes(), caption if index == 0 else None))
                except Exception:
                    LOGGER.exception("Reading activity photo failed")
            if items:
                self._send_album(chat_id, items)
                return
        # No photos (or no album sender) — just the text.
        self._notify(chat_id, summary)
