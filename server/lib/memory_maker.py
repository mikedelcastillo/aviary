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
from lib.clock import now_ph
from lib.find import currently_visible
from lib.imaging import downscale_jpeg
from lib.journal import MemoryEntry, append_entry
from lib.labels import pretty


LOGGER = logging.getLogger("lib.memory_maker")

SUMMARY_TIMEOUT_SECONDS = 60.0
# Telegram caps a photo/album caption at 1024 chars.
CAPTION_LIMIT = 1024


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _clip_caption(text: str) -> str:
    return text if len(text) <= CAPTION_LIMIT else text[: CAPTION_LIMIT - 1] + "…"


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
        night_mode: Callable[[], bool] | None = None,
        night_slowdown: float = 4.0,
        night_move_threshold: float = 12.0,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = now_ph,
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
        # When every camera is in night/IR we assume the birds (and we) are
        # asleep: report only on a major change, refresh on a much slower beat,
        # and report immediately the moment a camera leaves IR (a light / dawn).
        self._night_mode = night_mode
        self._night_slowdown = max(1.0, night_slowdown)
        self._night_move_threshold = night_move_threshold
        self._was_night = False
        self._clock = clock
        self._now = now
        self._reported_set: frozenset[str] = frozenset()
        self._last_report_at = clock()
        self._activity_since: datetime | None = None
        self._last_summary = ""
        self._activity_msgs: dict[str, int] = {}
        # Whether the last tracked message is an album/photo (edit its caption) or
        # plain text (edit its text).
        self._last_is_caption = False

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
        rows = self._registry.snapshot()
        visible = currently_visible(rows, self._fresh)
        visible_set = frozenset(visible)
        new_birds = visible_set - self._reported_set

        night = self._night_mode() if self._night_mode is not None else False
        woke = self._was_night and not night  # a camera left IR — light/dawn
        self._was_night = night
        interval = self._interval * (self._night_slowdown if night else 1.0)
        due = (now - self._last_report_at) >= interval

        # The moment the room lights up (or dawn breaks), report what's going on.
        if woke and visible:
            if self._report(visible):
                self._last_report_at = now
            return

        if night:
            # Assume resting: only break the quiet for a NEW bird or real motion;
            # otherwise just refresh ("still quiet") on the slow night beat.
            major = bool(new_birds) or self._has_motion(rows)
            if major and visible:
                if self._report(visible):
                    self._last_report_at = now
            elif due:
                self._refresh(visible_set)
                self._last_report_at = now
            return

        if new_birds or (due and visible_set != self._reported_set and visible_set):
            if self._report(visible):
                self._last_report_at = now
        elif due:
            self._refresh(visible_set)
            self._last_report_at = now

    def _has_motion(self, rows: list[dict]) -> bool:
        """True if a freshly-seen bird has moved notably — a major night event."""
        for row in rows:
            since = row.get("since")
            if since is None or since > self._fresh:
                continue
            if (row.get("movement_percent") or 0.0) >= self._night_move_threshold:
                return True
        return False

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

        # 1) Grab + save the frames.
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

        # 2) Describe each frame (VLM), summarise, remember.
        observations = []
        for image, camera, birds in shots:
            try:
                note = self._describe_frame(image) if self._describe_frame else None
            except Exception:
                LOGGER.exception("Memory describe failed")
                note = None
            who = ", ".join(pretty(b) for b in birds)
            observations.append(f"{who} on {self._camera_display(camera)}: {note or 'seen'}")
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

        # 3) Send it as ONE album — photos grouped, summary as the first caption —
        #    instead of N separate photos plus a text (which spammed the chat).
        self._activity_since = when
        self._last_summary = summary
        self._reported_set = frozenset(visible)
        header = f"🐦 {when.strftime('%H:%M')} — " + ", ".join(pretty(b) for b in all_birds)
        caption = _clip_caption(f"{header}\n{summary}".strip())
        items = [(img, caption if i == 0 else None) for i, (img, _, _) in enumerate(shots)]
        self._activity_msgs = self._notifier.broadcast_album_tracked(items)
        self._last_is_caption = True
        if not self._activity_msgs:
            # Album delivery failed for everyone; fall back to a tracked text so
            # the refresh path still has something to edit.
            self._activity_msgs = self._notifier.broadcast_text_tracked(f"{header}\n{summary}".strip())
            self._last_is_caption = False
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
            if self._last_is_caption:
                self._notifier.edit_message_caption(user_id, message_id, _clip_caption(body))
            else:
                self._notifier.edit_message_text(user_id, message_id, body)

    def _last_message_base(self) -> str:
        if self._activity_since and self._last_summary:
            return f"🐦 {self._activity_since.strftime('%H:%M')}\n{self._last_summary}"
        return self._last_summary or "🐦 Activity"
