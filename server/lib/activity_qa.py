"""Answer natural questions about what the birds did, from the activity log.

"what did percy do today?", "what is draft up to?", "what are the birds doing
right now?" — these read the collected-photos activity log (:mod:`lib.activity`),
caption the relevant photos with the vision model and summarise. If a specific
bird hasn't been seen recently and the question is about *now*, it automatically
kicks off a live ``/find`` (with its photo + description) instead of shrugging.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

from lib.activity import caption_sighting, load_sightings, select_highlights, summarise_day
from lib.find import pretty_phrase
from lib.labels import pretty
from lib.roster import expand_targets


LOGGER = logging.getLogger("lib.activity_qa")

# Default look-back for a "right now / lately" question (no explicit "today").
RECENT_WINDOW_SECONDS = 3 * 3600
# How many photos to caption for a Q&A answer (kept small so it stays snappy).
MAX_QA_PHOTOS = 3
CAPTION_TIMEOUT_SECONDS = 90.0
SUMMARY_TIMEOUT_SECONDS = 60.0

# Words that make an empty result trigger a live search instead of "haven't seen".
_LIVE_WORDS = ("now", "right now", "currently", "doing", "up to", "where")


class ActivityResponder:
    def __init__(
        self,
        collect_dir,
        client,
        llm_model: str,
        vlm_model: str,
        known_labels: Callable[[], list[str]],
        *,
        notify: Callable[[int, str], None],
        send_photo: Callable[[int, bytes, str | None], object] | None = None,
        find: Callable[[int, str], None] | None = None,
        member_species: dict[str, str] | None = None,
        pronouns: dict[str, str] | None = None,
        camera_display: Callable[[str], str] = lambda name: name,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._collect_dir = collect_dir
        self._client = client
        self._llm_model = llm_model
        self._vlm_model = vlm_model
        self._known_labels = known_labels
        self._notify = notify
        self._send_photo = send_photo
        self._find = find
        self._member_species = member_species or {}
        self._pronouns = pronouns or {}
        self._camera_display = camera_display
        self._clock = clock

    def _window_start(self, text: str, now: float) -> tuple[float, str]:
        if "today" in text.lower():
            midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
            return midnight.timestamp(), "today"
        return now - RECENT_WINDOW_SECONDS, "in the last few hours"

    def respond(self, chat_id: int, text: str, argument: str) -> None:
        now = self._clock()
        since, window_phrase = self._window_start(text, now)
        labels = self._known_labels()
        targets = expand_targets(argument, labels) if argument.strip() else None

        sightings = load_sightings(self._collect_dir, since, now)
        if targets is not None:
            target_set = set(targets)
            sightings = [s for s in sightings if s.label in target_set]

        if not sightings:
            who = pretty_phrase(argument) if argument.strip() else "the birds"
            # A live, specific "what is X doing now?" -> go look instead of shrug.
            wants_live = any(word in text.lower() for word in _LIVE_WORDS)
            if targets and self._find is not None and wants_live:
                self._notify(chat_id, f"I haven't seen {who} {window_phrase} — let me check the cameras…")
                self._find(chat_id, argument)
                return
            self._notify(chat_id, f"I haven't seen {who} {window_phrase}.")
            return

        highlights = select_highlights(sightings, MAX_QA_PHOTOS)
        observations: list[str] = []
        photos: list[tuple[bytes, str | None]] = []
        for sighting in highlights:
            try:
                caption = caption_sighting(
                    self._client, self._vlm_model, sighting, self._member_species,
                    self._pronouns, timeout_seconds=CAPTION_TIMEOUT_SECONDS,
                )
            except Exception:
                LOGGER.exception("QA caption failed")
                caption = ""
            where = self._camera_display(sighting.camera)
            observations.append(f"{pretty(sighting.label)} ({where}): {caption}".strip())
            try:
                photos.append((sighting.path.read_bytes(), f"{pretty(sighting.label)} — {where}"))
            except Exception:
                LOGGER.exception("Reading QA photo failed")

        try:
            summary = summarise_day(
                self._client, self._llm_model, observations,
                header=f"The user asked: {text}", timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
            )
        except Exception:
            LOGGER.exception("QA summary failed")
            summary = "; ".join(observations)

        self._notify(chat_id, summary or "; ".join(observations))
        if self._send_photo is not None:
            for image, caption in photos[:MAX_QA_PHOTOS]:
                try:
                    self._send_photo(chat_id, image, caption)
                except Exception:
                    LOGGER.exception("Sending QA photo failed")
