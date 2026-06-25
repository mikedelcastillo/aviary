"""The caretaker's running activity reports + persistent day memory.

Replaces the old fixed-cadence digest with something smarter:

  * It polls what the cameras see every ~30s. When a NEW bird appears (something
    worth telling you about), it reports straight away.
  * On a steady ~5-minute beat, if the same birds are still around it EDITS the
    last activity message in place ("still going — last updated 14:37") instead
    of sending a new one, and if it's quiet it just notes that. No spam.
  * Every report is also appended to the day's Markdown memory
    (``data/server/memories/YYYY-MM-DD.md``) so ``/activity`` and "what did percy
    do today?" can read it back, and it survives restarts.

Photo selection, captioning and summarising are reused from :mod:`lib.activity`;
the journal from :mod:`lib.journal`.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from lib.activity import caption_sighting, load_sightings, select_highlights, summarise_day
from lib.find import currently_visible
from lib.journal import MemoryEntry, append_entry
from lib.labels import pretty


LOGGER = logging.getLogger("lib.memory_maker")

CAPTION_TIMEOUT_SECONDS = 120.0
SUMMARY_TIMEOUT_SECONDS = 60.0


class MemoryMaker:
    def __init__(
        self,
        collect_dir: Path,
        memories_dir: Path,
        registry,
        client,
        llm_model: str,
        vlm_model: str,
        notifier,
        stop_event: threading.Event,
        *,
        interval_seconds: float = 300.0,
        poll_seconds: float = 30.0,
        fresh_seconds: float = 15.0,
        member_species: dict[str, str] | None = None,
        pronouns: dict[str, str] | None = None,
        camera_display: Callable[[str], str] = lambda name: name,
        max_photos: int = 4,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._collect_dir = collect_dir
        self._memories_dir = memories_dir
        self._registry = registry
        self._client = client
        self._llm_model = llm_model
        self._vlm_model = vlm_model
        self._notifier = notifier
        self._stop = stop_event
        self._interval = interval_seconds
        self._poll = poll_seconds
        self._fresh = fresh_seconds
        self._member_species = member_species or {}
        self._pronouns = pronouns or {}
        self._camera_display = camera_display
        self._max_photos = max_photos
        self._clock = clock
        self._now = now
        # State.
        self._reported_set: frozenset[str] = frozenset()
        self._last_report_at = clock()
        self._activity_since: datetime | None = None
        self._last_summary = ""
        self._activity_msgs: dict[str, int] = {}
        self._window_start = now()

    def start(self) -> None:
        threading.Thread(target=self._run, name="memory-maker", daemon=True).start()

    def _run(self) -> None:
        self._last_report_at = self._clock()
        # Anchor the first window to wall time so the first report covers a real
        # span of collected activity.
        self._window_start = self._now()
        LOGGER.info("Memory maker started (poll %.0fs, report beat %.0fs)", self._poll, self._interval)
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                break
            try:
                self._tick()
            except Exception:
                LOGGER.exception("Memory maker tick failed")

    def _tick(self) -> None:
        now = self._clock()
        visible = currently_visible(self._registry.snapshot(), self._fresh)
        visible_set = frozenset(visible)
        new_birds = visible_set - self._reported_set
        due = (now - self._last_report_at) >= self._interval

        if new_birds or (due and visible_set != self._reported_set and visible_set):
            self._report(visible_set)
            self._last_report_at = now
        elif due:
            self._refresh(visible_set)
            self._last_report_at = now

    # -- reporting ---------------------------------------------------------

    def _report(self, visible_set: frozenset[str]) -> None:
        until = self._now()
        sightings = load_sightings(self._collect_dir, self._window_start.timestamp(), until.timestamp())
        if not sightings:
            # A bird is visible but nothing collected yet; wait for the next tick.
            return
        highlights = select_highlights(sightings, self._max_photos)
        observations = []
        for sighting in highlights:
            try:
                caption = caption_sighting(
                    self._client, self._vlm_model, sighting, self._member_species,
                    self._pronouns, timeout_seconds=CAPTION_TIMEOUT_SECONDS,
                )
            except Exception:
                LOGGER.exception("Memory caption failed")
                caption = ""
            where = self._camera_display(sighting.camera)
            observations.append(f"{pretty(sighting.label)} ({where}): {caption}".strip())
        try:
            summary = summarise_day(
                self._client, self._llm_model, observations, timeout_seconds=SUMMARY_TIMEOUT_SECONDS
            )
        except Exception:
            LOGGER.exception("Memory summary failed")
            summary = "; ".join(observations)

        birds = sorted({s.label for s in sightings})
        primary_photo = str(highlights[0].path) if highlights else None
        try:
            append_entry(self._memories_dir, MemoryEntry(until, birds, summary or "(activity)", primary_photo))
        except Exception:
            LOGGER.exception("Writing memory entry failed")

        self._activity_since = until
        self._last_summary = summary
        self._window_start = until  # next report window starts here
        self._reported_set = visible_set

        # Photos first (reliable, individual), then the editable summary text.
        for sighting in highlights:
            try:
                image = sighting.path.read_bytes()
            except Exception:
                continue
            for user_id in self._notifier.user_ids:
                self._notifier.send_photo(
                    user_id, image, f"{pretty(sighting.label)} — {self._camera_display(sighting.camera)}"
                )
        header = f"🐦 {until.strftime('%H:%M')} — " + ", ".join(pretty(b) for b in birds)
        self._activity_msgs = self._notifier.broadcast_text_tracked(f"{header}\n{summary}".strip())
        LOGGER.info("Memory report sent (%d photo(s), birds=%s)", len(highlights), ",".join(birds))

    def _refresh(self, visible_set: frozenset[str]) -> None:
        """Edit the last activity message in place rather than sending a new one."""
        stamp = self._now().strftime("%H:%M")
        if visible_set:
            note = f"\n(Still the same — last updated {stamp}.)"
            body = f"{self._last_message_base()}{note}"
        else:
            since = self._activity_since.strftime("%H:%M") if self._activity_since else stamp
            body = f"😴 All quiet since {since}. Last checked {stamp}."
        if not self._activity_msgs:
            return
        for user_id, message_id in list(self._activity_msgs.items()):
            self._notifier.edit_message_text(user_id, message_id, body)

    def _last_message_base(self) -> str:
        if self._activity_since and self._last_summary:
            header = f"🐦 {self._activity_since.strftime('%H:%M')}"
            return f"{header}\n{self._last_summary}"
        return self._last_summary or "🐦 Activity"
