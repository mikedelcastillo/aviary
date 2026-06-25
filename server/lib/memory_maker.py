"""The caretaker's running activity reports + persistent day memory.

Self-contained: it grabs frames straight from the live cameras (not the
collect/filter tree, which may be retired), saves them into the memory image
store at ``data/server/memories/images/`` so they can be looked back on, runs the
vision model on them, and appends a timestamped entry — with the photos — to the
day's Markdown memory.

Behaviour:
  * Polls what the cameras see every ~30s. When a NEW bird appears, it reports
    straight away (photos first, summary follows).
  * On a steady ~5-minute beat, if the same birds are still around it EDITS the
    last activity message in place ("still the same — last updated 14:37"), and
    if it's quiet just notes that. No spam.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from lib.activity import summarise_activity
from lib.find import currently_visible
from lib.imaging import downscale_jpeg
from lib.journal import MemoryEntry, append_entry
from lib.labels import pretty


LOGGER = logging.getLogger("lib.memory_maker")

SUMMARY_TIMEOUT_SECONDS = 60.0


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class MemoryMaker:
    def __init__(
        self,
        memories_dir: Path,
        registry,
        grab_frame: Callable[[str], bytes | None],
        describe_frame: Callable[[bytes], str | None],
        client,
        llm_model: str,
        notifier,
        stop_event: threading.Event,
        *,
        interval_seconds: float = 300.0,
        poll_seconds: float = 30.0,
        fresh_seconds: float = 15.0,
        camera_display: Callable[[str], str] = lambda name: name,
        pronoun_note: str = "",
        max_cameras: int = 3,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._memories_dir = Path(memories_dir)
        self._images_dir = self._memories_dir / "images"
        self._registry = registry
        self._grab_frame = grab_frame
        self._describe_frame = describe_frame
        self._client = client
        self._llm_model = llm_model
        self._notifier = notifier
        self._stop = stop_event
        self._interval = interval_seconds
        self._poll = poll_seconds
        self._fresh = fresh_seconds
        self._camera_display = camera_display
        self._pronoun_note = pronoun_note
        self._max_cameras = max_cameras
        self._clock = clock
        self._now = now
        self._reported_set: frozenset[str] = frozenset()
        self._last_report_at = clock()
        self._activity_since: datetime | None = None
        self._last_summary = ""
        self._activity_msgs: dict[str, int] = {}

    def start(self) -> None:
        threading.Thread(target=self._run, name="memory-maker", daemon=True).start()

    def _run(self) -> None:
        self._last_report_at = self._clock()
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
            if self._report(visible):
                self._last_report_at = now
        elif due:
            self._refresh(visible_set)
            self._last_report_at = now

    # -- reporting ---------------------------------------------------------

    def _save_image(self, image: bytes, when: datetime, camera: str) -> Path:
        self._images_dir.mkdir(parents=True, exist_ok=True)
        path = self._images_dir / f"{when.strftime('%Y%m%d_%H%M%S')}_{_safe(camera)}.jpg"
        path.write_bytes(image)
        return path

    def _report(self, visible: dict[str, list[str]]) -> bool:
        """Capture live frames for the visible birds, save + describe + remember.

        Returns True if a report was actually sent.
        """
        cam_birds: dict[str, list[str]] = {}
        for label, cameras in visible.items():
            for camera in cameras:
                cam_birds.setdefault(camera, []).append(label)
        if not cam_birds:
            return False
        cameras = sorted(cam_birds, key=lambda c: (-len(cam_birds[c]), c))[: self._max_cameras]
        when = self._now()

        # 1) Grab + save + send the photos first (fast, reliable).
        shots: list[tuple[bytes, str, list[str]]] = []  # (image, camera, birds)
        saved_paths: list[str] = []
        for camera in cameras:
            try:
                image = self._grab_frame(camera)
            except Exception:
                LOGGER.exception("Memory grab_frame failed for %s", camera)
                image = None
            if not image:
                continue
            small = downscale_jpeg(image)
            try:
                saved_paths.append(str(self._save_image(small, when, camera)))
            except Exception:
                LOGGER.exception("Saving memory image failed")
            birds = sorted(set(cam_birds[camera]))
            shots.append((small, camera, birds))
        if not shots:
            return False

        for image, camera, birds in shots:
            caption = f"{', '.join(pretty(b) for b in birds)} — {self._camera_display(camera)}"
            for user_id in self._notifier.user_ids:
                self._notifier.send_photo(user_id, image, caption)

        # 2) Describe each frame (VLM, slower), summarise, remember, post the text.
        observations = []
        for image, camera, birds in shots:
            try:
                note = self._describe_frame(image) if self._describe_frame else None
            except Exception:
                LOGGER.exception("Memory describe failed")
                note = None
            who = ", ".join(pretty(b) for b in birds)
            observations.append(f"{who} ({self._camera_display(camera)}): {note or 'seen'}")
        try:
            summary = summarise_activity(
                self._client, self._llm_model, observations,
                pronoun_note=self._pronoun_note, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
            )
        except Exception:
            LOGGER.exception("Memory summary failed")
            summary = "; ".join(observations)

        all_birds = sorted({b for birds in cam_birds.values() for b in birds})
        try:
            append_entry(self._memories_dir, MemoryEntry(when, all_birds, summary or "(activity)", saved_paths))
        except Exception:
            LOGGER.exception("Writing memory entry failed")

        self._activity_since = when
        self._last_summary = summary
        self._reported_set = frozenset(visible)
        header = f"🐦 {when.strftime('%H:%M')} — " + ", ".join(pretty(b) for b in all_birds)
        self._activity_msgs = self._notifier.broadcast_text_tracked(f"{header}\n{summary}".strip())
        LOGGER.info("Memory report sent (%d photo(s), birds=%s)", len(shots), ",".join(all_birds))
        return True

    def _refresh(self, visible_set: frozenset[str]) -> None:
        """Edit the last activity message in place rather than sending a new one."""
        stamp = self._now().strftime("%H:%M")
        if visible_set:
            body = f"{self._last_message_base()}\n(Still the same — last updated {stamp}.)"
        else:
            since = self._activity_since.strftime("%H:%M") if self._activity_since else stamp
            body = f"😴 All quiet since {since}. Last checked {stamp}."
        for user_id, message_id in list(self._activity_msgs.items()):
            self._notifier.edit_message_text(user_id, message_id, body)

    def _last_message_base(self) -> str:
        if self._activity_since and self._last_summary:
            return f"🐦 {self._activity_since.strftime('%H:%M')}\n{self._last_summary}"
        return self._last_summary or "🐦 Activity"
