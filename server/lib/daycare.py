"""The daycare digest: periodic "wave" updates instead of per-detection spam.

Rather than firing a photo every time a bird trips the detector (which floods
the chat), a background narrator wakes every ``interval`` minutes, looks at what
was collected since the last wave, picks the best handful of photos, captions
them with the vision model and writes a warm caretaker summary with the language
model — then sends ONE album + summary to everyone. Quiet stretches are handled
gracefully: empty windows are skipped, but after a few in a row it sends a brief
"all quiet, the birds are resting" so nobody's left wondering.

The selection/reading lives in :mod:`lib.activity`; this module is the schedule,
the compose step, and the send.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from lib.activity import (
    caption_sighting,
    load_sightings,
    select_highlights,
    summarise_counts,
    summarise_day,
)
from lib.labels import pretty


LOGGER = logging.getLogger("lib.daycare")

# How many empty windows in a row before a gentle "all quiet" reassurance.
QUIET_REASSURE_AFTER = 3
# Per-call VLM/LLM timeouts for the digest (background, so generous).
CAPTION_TIMEOUT_SECONDS = 90.0
SUMMARY_TIMEOUT_SECONDS = 60.0


class DaycareNarrator:
    def __init__(
        self,
        collect_dir: Path,
        client,
        llm_model: str,
        vlm_model: str,
        notifier,
        stop_event: threading.Event,
        *,
        interval_seconds: float,
        member_species: dict[str, str] | None = None,
        camera_display: Callable[[str], str] = lambda name: name,
        max_photos: int = 6,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._collect_dir = collect_dir
        self._client = client
        self._llm_model = llm_model
        self._vlm_model = vlm_model
        self._notifier = notifier
        self._stop_event = stop_event
        self._interval = interval_seconds
        self._member_species = member_species or {}
        self._camera_display = camera_display
        self._max_photos = max_photos
        self._clock = clock
        self._thread: threading.Thread | None = None
        self._empty_streak = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="daycare-digest", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_ts = self._clock()
        LOGGER.info("Daycare digest started (every %.0f min)", self._interval / 60)
        while not self._stop_event.is_set():
            # Wait first, so the initial digest covers a real window of activity.
            if self._stop_event.wait(self._interval):
                break
            now = self._clock()
            try:
                self.run_digest(last_ts, now)
            except Exception:
                LOGGER.exception("Daycare digest failed")
            last_ts = now

    def run_digest(self, since: float, until: float) -> bool:
        """Compose and send one digest for ``[since, until]``. Returns True if sent."""
        sightings = load_sightings(self._collect_dir, since, until)
        if not sightings:
            self._empty_streak += 1
            if self._empty_streak >= QUIET_REASSURE_AFTER:
                self._empty_streak = 0
                self._notifier.broadcast_text(
                    "😴 All quiet at the aviary — the birds are resting. I'll ping you when they're up to something."
                )
            return False
        self._empty_streak = 0

        highlights = select_highlights(sightings, self._max_photos)
        observations: list[str] = []
        items: list[tuple[bytes, str | None]] = []
        for sighting in highlights:
            try:
                caption = caption_sighting(
                    self._client,
                    self._vlm_model,
                    sighting,
                    self._member_species,
                    timeout_seconds=CAPTION_TIMEOUT_SECONDS,
                )
            except Exception:
                LOGGER.exception("Caption failed for %s", sighting.path.name)
                caption = ""
            where = self._camera_display(sighting.camera)
            observations.append(f"{pretty(sighting.label)} ({where}): {caption}".strip())
            try:
                items.append((sighting.path.read_bytes(), f"{pretty(sighting.label)} — {where}"))
            except Exception:
                LOGGER.exception("Reading digest photo %s failed", sighting.path)

        try:
            summary = summarise_day(
                self._client, self._llm_model, observations, timeout_seconds=SUMMARY_TIMEOUT_SECONDS
            )
        except Exception:
            LOGGER.exception("Digest summary failed")
            summary = ""

        lead = f"☀️ Daycare update — seen recently: {summarise_counts(sightings)}."
        message = f"{lead}\n\n{summary}" if summary else lead
        self._notifier.broadcast_text(message)
        if items:
            self._notifier.broadcast_album(items)
        LOGGER.info("Sent daycare digest (%d photo(s), %d sighting(s))", len(items), len(sightings))
        return True
