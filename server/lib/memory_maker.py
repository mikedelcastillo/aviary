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
from lib.ai.client import OllamaBusyError, OllamaUnavailableError
from lib.clock import now_ph
from lib.find import currently_visible
from lib.imaging import downscale_jpeg
from lib.journal import MemoryEntry, MemoryObservation, append_entry
from lib.memory_build import annotate, build_observation
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
        client,
        llm_model: str,
        notifier,
        stop_event: threading.Event,
        *,
        detect_frame: Callable[[bytes], list] | None = None,
        analyze: Callable[[bytes, list], dict] | None = None,
        detector_model: str = "",
        vlm_model: str = "",
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
        self._detect_frame = detect_frame
        self._analyze = analyze
        self._detector_model = detector_model
        self._vlm_model = vlm_model
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

        # 1) Grab each frame, detect birds ON it, draw labeled boxes, save the
        #    annotated shot. Detecting the freshly-grabbed frame (rather than reusing
        #    the tracker's `visible` state) means the boxes, the stored detections and
        #    the saved photo all agree, and the birds we record are the ones actually
        #    in THIS frame.
        shots: list[tuple[bytes, str, list[str], str, list]] = []
        for camera in cameras:
            try:
                image = self._grab_frame(camera)
            except Exception:
                LOGGER.exception("Memory grab_frame failed for %s", camera)
                image = None
            if not image:
                continue
            small = downscale_jpeg(image)
            detections: list = []
            if self._detect_frame is not None:
                try:
                    detections = self._detect_frame(small) or []
                except Exception:
                    LOGGER.exception("Memory detect failed for %s", camera)
            annotated = annotate(small, detections)
            # The frame's own detections are the truth for what's in the saved photo;
            # fall back to the trigger's labels only when detection is unavailable.
            birds = sorted({d.label for d in detections}) or sorted(set(cam_birds[camera]))
            # Save the RAW frame (no boxes) — the stored bboxes let boxes be redrawn
            # any time, so the memory keeps a clean, re-annotatable original. The boxed
            # `annotated` is used only for the VLM and the Telegram album.
            saved_path = ""
            try:
                saved_path = str(self._save_image(small, when, camera))
            except Exception:
                LOGGER.exception("Saving memory image failed")
            shots.append((annotated, camera, birds, saved_path, detections))
        if not shots:
            return False

        # 2) Structured VLM analysis of each ANNOTATED frame (it sees the labeled
        #    boxes, so activity is attributed to the right bird), then build the v3
        #    per-bird observation. YOLO stays authoritative for identity.
        raw_observations: list[str] = []
        structured_observations: list[MemoryObservation] = []
        for image, camera, birds, saved_path, detections in shots:
            analysis: dict | None = None
            if self._analyze is not None and detections:
                try:
                    analysis = self._analyze(image, birds)
                except (OllamaUnavailableError, OllamaBusyError):
                    # The entry is still written (identity + photo) with NO
                    # vlm_model stamp — the backfill worker decorates it from the
                    # saved photo once the cluster is back / has room.
                    LOGGER.warning("Memory analyze skipped: Ollama down/busy (will backfill)")
                except Exception:
                    LOGGER.exception("Memory analyze failed")
            display_camera = self._camera_display(camera)
            obs = build_observation(
                detections,
                camera=display_camera,
                photo=saved_path,
                analysis=analysis,
                detector_model=self._detector_model,
                vlm_model=self._vlm_model,
            )
            if not obs.detections:  # no detector hits — keep the trigger birds + a stub
                obs.birds = birds
            who = ", ".join(pretty(b) for b in (obs.birds or birds))
            raw_observations.append(f"{who} on {display_camera}: {obs.note or 'seen'}")
            structured_observations.append(obs)
        try:
            summary = summarise_activity(
                self._client, self._llm_model, raw_observations,
                pronoun_note=self._pronoun_note, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
            )
        except OllamaUnavailableError:
            LOGGER.warning("Memory summary skipped: Ollama down")
            summary = "; ".join(raw_observations)
        except Exception:
            LOGGER.exception("Memory summary failed")
            summary = "; ".join(raw_observations)

        # Only the birds we actually captured a frame for — a bird that was
        # visible on a camera that couldn't produce a frame is NOT reported (so
        # it stays "new" and is retried), nor claimed in the memory entry.
        captured = {bird for shot in shots for bird in shot[2]}
        all_birds = sorted(captured)
        saved_paths = [shot[3] for shot in shots if shot[3]]
        journal_note = (
            raw_observations[0]
            if len(raw_observations) == 1
            else "\n".join(f"- {line}" for line in raw_observations)
        )
        try:
            append_entry(
                self._memories_dir,
                MemoryEntry(
                    when,
                    all_birds,
                    journal_note or summary or "(activity)",
                    saved_paths,
                    observations=structured_observations,
                ),
            )
        except Exception:
            LOGGER.exception("Writing memory entry failed")

        # 3) Send it as ONE album — photos grouped, summary as the first caption —
        #    instead of N separate photos plus a text (which spammed the chat).
        self._activity_since = when
        self._last_summary = summary
        self._reported_set = frozenset(captured)
        header = f"🐦 {when.strftime('%H:%M')} — " + ", ".join(pretty(b) for b in all_birds)
        caption = _clip_caption(f"{header}\n{summary}".strip())
        items = [(shot[0], caption if i == 0 else None) for i, shot in enumerate(shots)]
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
