"""Locate one or more birds across every camera (the ``/find`` command).

"find percy", "find the cockatiels", "find any bird" — the target is resolved
(via :mod:`lib.roster`) into a SET of detector labels, and the search succeeds
the moment ANY of them turns up. It watches the shared
:class:`~lib.objects.ObjectRegistry` that every camera thread keeps updated, and
while it searches it nudges the pan-tilt cameras through a patrol sweep
(best-effort; see :mod:`lib.ptz`). On a hit it sends photo proof and — when a
vision model is wired — a short description of what the bird is doing and who
it's with.

The search runs on its OWN thread so a five-minute hunt never blocks the
Telegram poll loop. Progress is context-aware: a message only when the set of
visible birds changes, plus an occasional heartbeat.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from lib.labels import pretty
from lib.roster import DEFAULT_SPECIES_MEMBERS, expand_targets


LOGGER = logging.getLogger("lib.find")

DEFAULT_FRESH_SECONDS = 12.0
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_TICK_SECONDS = 10.0
DEFAULT_HEARTBEAT_SECONDS = 60.0


class _Snapshotter(Protocol):
    def snapshot(self) -> list[dict]: ...


def pretty_phrase(text: str) -> str:
    """Title-case a free-text request: "cockatiels" -> "Cockatiels"."""
    return text.strip().title() if text.strip() else "that bird"


def short_camera(name: str) -> str:
    """Trim ``camera-192.168.1.8`` to ``.8`` for terse, readable updates."""
    host = name.removeprefix("camera-")
    if host.count(".") == 3:
        return "." + host.rsplit(".", 1)[1]
    return name


def currently_visible(rows: list[dict], fresh_seconds: float) -> dict[str, list[str]]:
    """Map each freshly-seen label to the cameras seeing it (within ``fresh_seconds``)."""
    visible: dict[str, set[str]] = {}
    for row in rows:
        since = row.get("since")
        if since is None or since > fresh_seconds:
            continue
        visible.setdefault(str(row["label"]).lower(), set()).add(str(row["camera"]))
    return {label: sorted(cameras) for label, cameras in visible.items()}


def format_visible(visible: dict[str, list[str]], display=short_camera) -> str:
    """Render the visible map as ``Percy (Window Perch), Matcha (Food Bowl)``."""
    if not visible:
        return "no birds in view"
    parts = []
    for label in sorted(visible):
        cams = ", ".join(display(camera) for camera in visible[label])
        parts.append(f"{pretty(label)} ({cams})")
    return "; ".join(parts)


def format_found_message(
    found_labels: list[str],
    visible: dict[str, list[str]],
    description: str | None = None,
    display=short_camera,
) -> str:
    parts = []
    for label in found_labels:
        cams = ", ".join(display(camera) for camera in visible.get(label, []))
        parts.append(f"{pretty(label)} on {cams}")
    message = f"🔎 Found {'; '.join(parts)}!"
    if description:
        message += f"\n{description}"
    return message


def format_progress_message(requested: str, visible: dict[str, list[str]], display=short_camera) -> str:
    return (
        f"Still looking for {pretty_phrase(requested)}. "
        f"Right now I can see: {format_visible(visible, display)}."
    )


def format_not_found_message(
    requested: str, elapsed: float, ever_seen: dict[str, list[str]], display=short_camera
) -> str:
    minutes = max(1, int(round(elapsed / 60)))
    base = (
        f"😕 Couldn't find {pretty_phrase(requested)} after {minutes} min. "
        "They may be out of frame, behind something, or resting."
    )
    if ever_seen:
        base += f" While searching I did see: {format_visible(ever_seen, display)}."
    return base


@dataclass
class FindOutcome:
    requested: str
    found: bool
    found_labels: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class BirdFinder:
    """Runs at most one search at a time, on a background thread."""

    def __init__(
        self,
        registry: _Snapshotter,
        known_labels: Callable[[], list[str]],
        *,
        notify: Callable[[int, str], None],
        grab_frame: "Callable[[str], bytes | None] | None" = None,
        send_album: "Callable[[int, Sequence[tuple[bytes, str | None]]], None] | None" = None,
        describe_frame: "Callable[[bytes], str | None] | None" = None,
        make_patrol: "Callable[[], PtzPatrol | None] | None" = None,
        camera_display: Callable[[str], str] = short_camera,
        species_members: dict[str, tuple[str, ...]] | None = None,
        fresh_seconds: float = DEFAULT_FRESH_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._known_labels = known_labels
        self._notify = notify
        self._grab_frame = grab_frame
        self._send_album = send_album
        # Optional VLM: given a frame's JPEG bytes, return a short scene
        # description ("Percy is on the perch with Matcha"). Best-effort.
        self._describe_frame = describe_frame
        self._make_patrol = make_patrol
        self._camera_display = camera_display
        self._species_members = species_members or DEFAULT_SPECIES_MEMBERS
        self._fresh_seconds = fresh_seconds
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds
        self._tick_seconds = tick_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._clock = clock
        self._active_lock = threading.Lock()
        # The one in-flight search, or None. A dict {token, requested, cancel}:
        # ``token`` identifies the search so its thread only clears state it still
        # owns (a replacement may have taken over); ``cancel`` is its private stop
        # signal, set by stop_current() or by a replacing search.
        self._active: dict | None = None

    # -- validation --------------------------------------------------------

    def findable_labels(self) -> list[str]:
        return self._known_labels()

    def resolve_targets(self, requested: str) -> list[str]:
        """Expand a free-text request into concrete findable labels."""
        return expand_targets(requested, self._known_labels(), self._species_members)

    def _options_text(self) -> str:
        birds = ", ".join(pretty(label) for label in self.findable_labels())
        return (
            f"I can look for: {birds or '(no labels)'} — or groups like the "
            "cockatiels, the lovebirds, or all the birds."
        )

    # -- entry point -------------------------------------------------------

    # Words that mean "stop the current search" rather than "search for X".
    STOP_WORDS = {"stop", "cancel", "stop looking", "cancel search", "stop searching"}

    def stop_current(self) -> str:
        """Cancel the in-flight search, if any. Returns a message for the user."""
        with self._active_lock:
            active = self._active
            if active is None:
                return "No search is running right now."
            active["cancel"].set()
            requested = active["requested"]
        return f"🛑 Stopped searching for {pretty_phrase(requested)}."

    def is_searching(self) -> bool:
        with self._active_lock:
            return self._active is not None

    def start(self, chat_id: int, requested: str, stop_event: threading.Event) -> str:
        """Validate and launch a search; return the immediate ack/refusal text.

        "stop"/"cancel" cancels the running search. Starting a new search while
        one is already running REPLACES it ("look for draft instead"): the old
        one is cancelled silently and the new one takes over.
        """
        if requested.strip().lower() in self.STOP_WORDS:
            return self.stop_current()
        if not requested.strip():
            return f"Usage: /find <bird>. {self._options_text()}"
        targets = self.resolve_targets(requested)
        if not targets:
            return f"I don't know \"{requested.strip()}\". {self._options_text()}"

        cancel = threading.Event()
        token = object()
        with self._active_lock:
            previous = self._active
            self._active = {"token": token, "requested": requested.strip(), "cancel": cancel}
        replacing = previous is not None
        if replacing:
            previous["cancel"].set()  # silently retire the old search

        thread = threading.Thread(
            target=self._run_guarded,
            args=(token, chat_id, requested.strip(), targets, stop_event, cancel),
            name="find",
            daemon=True,
        )
        thread.start()
        minutes = int(round(self._timeout_seconds / 60))
        whom = ", ".join(pretty(label) for label in targets[:4])
        if len(targets) > 4:
            whom += f" +{len(targets) - 4} more"
        lead = "🔄 Switching — now searching" if replacing else "🔭 On it — searching"
        return (
            f"{lead} all cameras for {whom} (up to {minutes} min). "
            "I'll ping you the moment I spot one. (Say \"stop looking\" to cancel.)"
        )

    def _run_guarded(self, token, chat_id, requested, targets, stop_event, cancel) -> None:
        try:
            self._run(chat_id, requested, targets, stop_event, cancel)
        except Exception:
            LOGGER.exception("Find loop failed for %r", requested)
            try:
                self._notify(chat_id, f"Search for {pretty_phrase(requested)} hit an error and stopped.")
            except Exception:
                LOGGER.exception("Failed to send find-error message")
        finally:
            # Only clear if we still own the slot — a replacement may have taken
            # over while we were winding down.
            with self._active_lock:
                if self._active is not None and self._active["token"] is token:
                    self._active = None

    # -- the search loop ---------------------------------------------------

    def _run(self, chat_id, requested, targets, stop_event, cancel) -> FindOutcome:
        patrol = None
        if self._make_patrol is not None:
            try:
                patrol = self._make_patrol()
            except Exception:
                LOGGER.exception("Building PTZ patrol failed; searching without it")
        try:
            return self._search(chat_id, requested, targets, stop_event, cancel, patrol)
        finally:
            if patrol is not None:
                try:
                    patrol.stop()
                except Exception:
                    LOGGER.exception("PTZ patrol stop failed")

    def _search(self, chat_id, requested, targets, stop_event, cancel, patrol) -> FindOutcome:
        start = self._clock()
        deadline = start + self._timeout_seconds
        next_tick = start + self._tick_seconds
        last_visible_keys: frozenset[str] | None = None
        last_message_at = start
        ever_seen: dict[str, list[str]] = {}

        if patrol is not None:
            try:
                patrol.start()
            except Exception:
                LOGGER.exception("PTZ patrol start failed; searching without it")

        while not stop_event.is_set() and not cancel.is_set() and self._clock() < deadline:
            visible = currently_visible(self._registry.snapshot(), self._fresh_seconds)
            for label, cams in visible.items():
                ever_seen[label] = sorted(set(ever_seen.get(label, [])) | set(cams))

            found_labels = [label for label in targets if label in visible]
            if found_labels:
                cameras = sorted({cam for label in found_labels for cam in visible[label]})
                # Announce + send the proof photo FIRST: it's fast and reliable,
                # and must not depend on the slow VLM. The scene description is a
                # best-effort follow-up so a slow/cold vision model never costs us
                # the photo.
                self._notify(
                    chat_id,
                    format_found_message(found_labels, visible, None, self._camera_display),
                )
                self._send_found_photos(chat_id, found_labels, visible)
                description = self._describe_camera(cameras[0]) if cameras else None
                if description:
                    self._notify(chat_id, f"📝 {description}")
                return FindOutcome(requested, True, found_labels, cameras, self._clock() - start)

            now = self._clock()
            if now >= next_tick:
                next_tick = now + self._tick_seconds
                keys = frozenset(visible)
                if keys != last_visible_keys or (now - last_message_at) >= self._heartbeat_seconds:
                    self._notify(
                        chat_id, format_progress_message(requested, visible, self._camera_display)
                    )
                    last_visible_keys = keys
                    last_message_at = now

            if patrol is not None:
                try:
                    patrol.step()
                except Exception:
                    LOGGER.exception("PTZ patrol step failed")

            # Wait on the cancel signal so "stop looking" / a replacement
            # interrupts the poll immediately instead of after the full interval.
            if cancel.wait(self._poll_seconds):
                break

        # Cancelled (explicit stop or replaced) and shutdown both exit quietly —
        # stop_current()/start() already messaged the user. Only a real timeout
        # sends the not-found recap.
        if cancel.is_set() or stop_event.is_set():
            return FindOutcome(requested, False, [], [], self._clock() - start)
        elapsed = self._clock() - start
        self._notify(
            chat_id, format_not_found_message(requested, elapsed, ever_seen, self._camera_display)
        )
        return FindOutcome(requested, False, [], [], elapsed)

    # -- vision + photo helpers -------------------------------------------

    def _describe_camera(self, camera: str) -> str | None:
        """Best-effort VLM description of what a camera currently shows."""
        if self._grab_frame is None or self._describe_frame is None:
            return None
        try:
            image = self._grab_frame(camera)
            if not image:
                return None
            return self._describe_frame(image) or None
        except Exception:
            LOGGER.exception("Scene description failed for %s", camera)
            return None

    def _send_found_photos(self, chat_id, found_labels, visible) -> int:
        """Send the latest frame of each camera that saw a found bird, as proof.

        Returns how many photos were sent (0 if vision/album wiring is absent or
        no camera had a current frame). Logged so a missing photo is diagnosable.
        """
        if self._grab_frame is None or self._send_album is None:
            LOGGER.warning("Find: no grab_frame/send_album wired; cannot send proof photo")
            return 0
        cameras = sorted({cam for label in found_labels for cam in visible.get(label, [])})
        items: list[tuple[bytes, str | None]] = []
        for camera in cameras:
            try:
                image = self._grab_frame(camera)
            except Exception:
                LOGGER.exception("Grabbing proof frame for %s failed", camera)
                image = None
            if not image:
                LOGGER.warning("Find: no current frame for %s; skipping its photo", camera)
                continue
            here = ", ".join(
                pretty(label) for label in found_labels if camera in visible.get(label, [])
            )
            items.append((image, f"{here} — {self._camera_display(camera)}"))
        if not items:
            return 0
        try:
            self._send_album(chat_id, items)
            LOGGER.info("Find: sent %d proof photo(s) to chat %s", len(items), chat_id)
            return len(items)
        except Exception:
            LOGGER.exception("Sending found photos failed")
            return 0


class PtzPatrol(Protocol):
    """Best-effort camera-movement driver used while a search runs."""

    def start(self) -> None: ...
    def step(self) -> None: ...
    def stop(self) -> None: ...
