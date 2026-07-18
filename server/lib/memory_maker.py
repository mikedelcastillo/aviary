"""The caretaker's running activity reports + persistent day memory.

Self-contained: it grabs frames straight from the live cameras (not the
collect/filter tree, which may be retired), saves them into the memory image
store at ``data/server/memories/images/`` so they can be looked back on, runs the
vision model on them, and appends a timestamped entry — with the photos — to the
day's Markdown memory.

Behaviour:
  * Polls what the cameras see every ~30s. When a NEW bird appears, it reports
    straight away (photos first, summary follows).
  * Every report is verified against the frame it sends: a bird the detector
    can't find on the freshly-grabbed frame is neither photographed nor claimed
    (no empty-floor photos captioned with birds), and only what the VLM actually
    saw reaches the summariser (stub sightings get a plain factual line, never
    an LLM guess).
  * On a steady ~5-minute beat, if the same birds are still around it EDITS the
    last activity message in place ("still the same — last updated 14:37"), and
    if it's quiet just notes that. No spam.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
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
# How many extra polls a failed wake-up report keeps retrying. Covers the
# camera's auto-exposure settling after IR flips off (a few seconds) without
# letting a registry-phantom bird churn the grab+detect pipeline forever.
WAKE_RETRY_LIMIT = 4


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _clip_caption(text: str) -> str:
    return text if len(text) <= CAPTION_LIMIT else text[: CAPTION_LIMIT - 1] + "…"


@dataclass
class _Shot:
    """One camera's verified frame within a report: the boxed copy for the
    album/VLM, the clean original for the memory store, and what's on it."""

    annotated: bytes
    camera: str
    birds: list[str]
    raw: bytes
    detections: list = field(default_factory=list)
    saved_path: str = ""


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
        # Birds whose report attempt couldn't confirm them on a frame, mapped to
        # the monotonic deadline before they may TRIGGER a report again. The
        # registry can keep such a bird "fresh" long after the frames say it's
        # gone; without this cooldown it re-triggers every poll, re-sending
        # near-identical albums for the birds that DID verify.
        self._drop_until: dict[str, float] = {}
        self._wake_retries = 0
        self._last_motion_report_at = float("-inf")
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
        cooled = {bird for bird, until in self._drop_until.items() if until > now}
        new_birds = visible_set - self._reported_set - cooled

        night = self._night_mode() if self._night_mode is not None else False
        woke = self._was_night and not night  # a camera left IR — light/dawn
        self._was_night = night
        interval = self._interval * (self._night_slowdown if night else 1.0)
        due = (now - self._last_report_at) >= interval

        # The moment the room lights up (or dawn breaks), report what's going on.
        if woke:
            if not visible:
                # The registry can be blind on the exact IR-flip tick (the
                # exposure swing blinds the live detector too). Consuming the
                # edge here would silently lose the dawn report — the birds are
                # usually already in _reported_set from the night, so no other
                # trigger would ever fire. Stay armed until there is something
                # to try; a wake retry is only counted for a failed ATTEMPT.
                self._was_night = True
            elif self._report(visible, allow_same=True):
                self._wake_retries = 0
                self._last_report_at = now
                return
            elif self._wake_retries < WAKE_RETRY_LIMIT:
                # Nothing verified on a frame (auto-exposure still settling,
                # birds mid-flap) — keep the wake edge armed and retry next
                # poll, but bounded: a registry-phantom bird must not churn
                # the grab+detect pipeline every poll forever.
                self._wake_retries += 1
                self._was_night = True
            else:
                self._wake_retries = 0  # give up; the normal beat takes over
            if due:
                # A failing wake retry must never freeze the tracked message.
                self._refresh(visible_set)
                self._last_report_at = now
            return

        if night:
            self._wake_retries = 0
            # Assume resting: only break the quiet for a NEW bird or real motion;
            # otherwise just refresh ("still quiet") on the slow night beat.
            # Motion re-reports the same birds on purpose (that's the event) but
            # at most once per base beat — a restless spell spans many polls and
            # must not re-send an album on every one of them. A new-bird trigger
            # must still confirm someone actually new.
            motion = (
                self._has_motion(rows)
                and now - self._last_motion_report_at >= self._interval
            )
            major = bool(new_birds) or motion
            if major and visible and self._report(visible, allow_same=motion):
                self._last_report_at = now
                if motion:
                    self._last_motion_report_at = now
            elif due:
                self._refresh(visible_set)
                self._last_report_at = now
            return

        if new_birds or (due and visible_set != self._reported_set and visible_set):
            if self._report(visible):
                self._last_report_at = now
                return
        if due:
            # Nothing to report, or the report produced nothing (no frame
            # verified) — consume the beat with an in-place refresh either way,
            # so the tracked message's "last updated" stamp never freezes while
            # an unverifiable bird stays registry-fresh.
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

    def _cooldown(self, birds: set[str]) -> None:
        """Bar ``birds`` from TRIGGERING a report for one beat.

        Applied to every trigger bird a report that SENT something could not
        confirm on a frame. The registry can keep such a bird "fresh" long
        after it moved on, and an eager per-poll retry re-sent near-identical
        albums (and duplicate journal entries) for the birds that DID verify.
        The due beat — and the wake-up path, which bypasses triggers — still
        retry it. One beat means the OPERATIVE beat: at night the report
        cadence slows by night_slowdown, and a shorter cooldown would let a
        phantom bird re-trigger several times per night beat.
        """
        night = self._night_mode() if self._night_mode is not None else False
        until = self._clock() + self._interval * (self._night_slowdown if night else 1.0)
        for bird in birds:
            self._drop_until[bird] = until

    def _save_image(self, image: bytes, when: datetime, camera: str) -> Path:
        self._images_dir.mkdir(parents=True, exist_ok=True)
        path = self._images_dir / f"{when.strftime('%Y%m%d_%H%M%S')}_{_safe(camera)}.jpg"
        path.write_bytes(image)
        return path

    def _report(self, visible: dict[str, list[str]], *, allow_same: bool = False) -> bool:
        """Capture live frames for the visible birds, save + describe + remember.

        Returns True if a report was actually sent. A bird that couldn't be
        confirmed on a frame (its shot verified empty, its camera's grab failed,
        or its camera didn't make the cut) is NOT reported or claimed in the
        memory entry. A report that confirms only birds ALREADY reported is
        skipped too (the ghost trigger cools down for a beat; the edit-in-place
        refresh owns "still the same") — unless ``allow_same`` is set, for the
        wake-up and night-motion reports whose entire point is re-showing the
        same flock.
        """
        cam_birds: dict[str, list[str]] = {}
        for label, cameras in visible.items():
            for camera in cameras:
                cam_birds.setdefault(camera, []).append(label)
        if not cam_birds:
            return False
        # A camera carrying a not-yet-reported bird must make the cut ahead of
        # busier cameras full of already-reported birds — otherwise a newcomer
        # on a quiet camera could be cut (and its report skipped) indefinitely.
        unreported = set(visible) - set(self._reported_set)
        cameras = sorted(
            cam_birds,
            key=lambda c: (-len(set(cam_birds[c]) & unreported), -len(cam_birds[c]), c),
        )[: self._max_cameras]
        when = self._now()

        # 1) Grab each frame and detect birds ON it. Detecting the freshly-grabbed
        #    frame (rather than reusing the tracker's `visible` state) means the
        #    boxes, the stored detections and the saved photo all agree, and the
        #    birds we record are the ones actually in THIS frame.
        shots: list[_Shot] = []
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
            verified = False
            if self._detect_frame is not None:
                try:
                    detections = self._detect_frame(small) or []
                    verified = True
                except Exception:
                    LOGGER.exception("Memory detect failed for %s", camera)
            if verified and not detections:
                # The trigger bird left between the registry sighting and this
                # grab — the frame verifiably shows no birds. Shipping it under
                # the stale trigger labels paired an empty photo with a caption
                # about birds, so drop the shot.
                LOGGER.info(
                    "Memory shot dropped: no birds detected on %s (trigger: %s)",
                    camera,
                    ",".join(sorted(set(cam_birds[camera]))),
                )
                continue
            # The frame's own detections are the truth for what's in the saved photo;
            # fall back to the trigger's labels only when detection is unavailable
            # (no detector wired, or the detect call itself failed).
            birds = sorted({d.label for d in detections}) or sorted(set(cam_birds[camera]))
            shots.append(_Shot(annotate(small, detections), camera, birds, small, detections))
        if not shots:
            # Nothing was sent or journaled, so a next-poll retry cannot
            # duplicate anything — no cooldown: a new bird's first report
            # shouldn't wait out a beat over one mid-flap grab. (A lone phantom
            # does re-run grab+detect each poll, but that's noise next to the
            # live pipeline's continuous per-frame detection.)
            return False

        # A report that confirms nothing beyond birds the user has already been
        # shown — while nobody has actually departed — is skipped: the in-place
        # refresh owns "still the same", and sending would duplicate the album
        # and journal entry. The unconfirmed triggers cool down until the next
        # beat. A genuine departure (a reported bird gone from the registry)
        # still sends, so the album's composition reconciles with reality.
        captured = {bird for shot in shots for bird in shot.birds}
        reported = set(self._reported_set)
        if not allow_same and captured <= reported and reported <= set(visible):
            self._cooldown(set(visible) - captured)
            LOGGER.info(
                "Memory report skipped: nothing new confirmed (already reported: %s)",
                ",".join(sorted(captured)),
            )
            return False

        # Save each RAW frame (no boxes) — the stored bboxes let boxes be redrawn
        # any time, so the memory keeps a clean, re-annotatable original. The boxed
        # `annotated` copy is used only for the VLM and the Telegram album.
        for shot in shots:
            try:
                shot.saved_path = str(self._save_image(shot.raw, when, shot.camera))
            except Exception:
                LOGGER.exception("Saving memory image failed")

        # 2) Structured VLM analysis of each ANNOTATED frame (it sees the labeled
        #    boxes, so activity is attributed to the right bird), then build the v3
        #    per-bird observation. YOLO stays authoritative for identity.
        raw_observations: list[str] = []
        grounded_notes: list[str] = []
        stub_lines: list[str] = []
        structured_observations: list[MemoryObservation] = []
        for shot in shots:
            analysis: dict | None = None
            if self._analyze is not None and shot.detections:
                try:
                    analysis = self._analyze(shot.annotated, shot.birds)
                except (OllamaUnavailableError, OllamaBusyError):
                    # The entry is still written (identity + photo) with NO
                    # vlm_model stamp — the backfill worker decorates it from the
                    # saved photo once the cluster is back / has room.
                    LOGGER.warning("Memory analyze skipped: Ollama down/busy (will backfill)")
                except Exception:
                    LOGGER.exception("Memory analyze failed")
            display_camera = self._camera_display(shot.camera)
            obs = build_observation(
                shot.detections,
                camera=display_camera,
                photo=shot.saved_path,
                analysis=analysis,
                detector_model=self._detector_model,
                vlm_model=self._vlm_model,
            )
            if not obs.detections:  # no detector hits — keep the trigger birds + a stub
                obs.birds = shot.birds
            who = ", ".join(pretty(b) for b in (obs.birds or shot.birds))
            # Ground the note in everything the VLM saw on THIS frame: the scene
            # plus each bird's activity (previously discarded, which left the
            # summariser with nothing but names to work from).
            details = "; ".join(
                f"{pretty(d.label)} {d.activity}" for d in obs.detections if d.activity
            )
            grounded = ", ".join(part for part in (obs.note, details) if part)
            raw_observations.append(f"{who} on {display_camera}: {grounded or 'seen'}")
            if grounded:
                grounded_notes.append(raw_observations[-1])
            else:
                # A contentless "seen" stub (VLM down/skipped). Fed to the
                # summariser these made it INVENT activity — and pull in absent
                # roster birds from the pronoun note — so a stub NEVER reaches
                # the LLM: it becomes a plain factual bullet, and the backfill
                # decorates the journal entry once the VLM is back.
                stub_lines.append(f"• {who} seen on {display_camera}.")
            structured_observations.append(obs)
        summary = ""
        if grounded_notes:
            try:
                summary = summarise_activity(
                    self._client, self._llm_model, grounded_notes,
                    pronoun_note=self._pronoun_note, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                )
            except OllamaUnavailableError:
                LOGGER.warning("Memory summary skipped: Ollama down")
            except Exception:
                LOGGER.exception("Memory summary failed")
            if not summary:
                summary = "; ".join(grounded_notes)
        summary = "\n".join(line for line in [summary, *stub_lines] if line)

        # Only the birds we actually captured a frame for are claimed; a trigger
        # bird this report couldn't confirm cools down until the next beat.
        self._cooldown(set(visible) - captured)
        all_birds = sorted(captured)
        saved_paths = [shot.saved_path for shot in shots if shot.saved_path]
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
        items = [(shot.annotated, caption if i == 0 else None) for i, shot in enumerate(shots)]
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
            suffix = f"\n(Still the same — last updated {stamp}.)"
            base = self._last_message_base()
            # Clip the BASE, never the stamp: clipping the joined text from the
            # end cuts the stamp — the only part that changes — so every edit
            # became byte-identical and Telegram rejected it as unmodified.
            if len(base) + len(suffix) > CAPTION_LIMIT:
                base = base[: CAPTION_LIMIT - len(suffix) - 1] + "…"
            body = base + suffix
        else:
            since = self._activity_since.strftime("%H:%M") if self._activity_since else stamp
            body = f"😴 All quiet since {since}. Last checked {stamp}."
            # The flock story ended: when birds return later — even the exact
            # same birds — that's a fresh report, not a "still the same" edit
            # on an hours-old album.
            self._reported_set = frozenset()
        for user_id, message_id in list(self._activity_msgs.items()):
            if self._last_is_caption:
                self._notifier.edit_message_caption(user_id, message_id, _clip_caption(body))
            else:
                self._notifier.edit_message_text(user_id, message_id, body)

    def _last_message_base(self) -> str:
        if self._activity_since and self._last_summary:
            return f"🐦 {self._activity_since.strftime('%H:%M')}\n{self._last_summary}"
        return self._last_summary or "🐦 Activity"
