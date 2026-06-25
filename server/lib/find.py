"""Locate a specific bird across every camera (the ``/find`` command).

``/find percy`` answers "where is Percy right now?". It watches the shared
:class:`~lib.objects.ObjectRegistry` — which every camera thread keeps updated
with what its detector last saw — and reports the moment the target turns up,
giving up after a timeout. While it searches it nudges any pan-tilt cameras
through a patrol sweep (best-effort; see :mod:`lib.ptz`) so a bird tucked out of
frame still gets found.

The search runs on its OWN thread, never on the Telegram poll loop: a five
minute hunt must not block /pause or /status. Progress is pushed to the
requesting chat via a callback. Updates are deliberately *context-aware* — a new
message only when the set of visible birds actually changes, plus an occasional
heartbeat — so a long search doesn't turn into a stream of identical texts.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


LOGGER = logging.getLogger("lib.find")

# A label counts as "currently visible" if a camera saw it within this window.
# Cameras sample ~1 fps, so a few sample intervals smooths over the frame the
# bird happened to hop out of without going stale.
DEFAULT_FRESH_SECONDS = 12.0
# Total time to hunt before giving up, per the spec ("timeout of 5 minutes").
DEFAULT_TIMEOUT_SECONDS = 300.0
# How often to re-check the registry (and step the PTZ patrol).
DEFAULT_POLL_SECONDS = 2.0
# Cadence of the "every 10 seconds" progress check. A message is only actually
# sent when the visible set changed or the heartbeat below is due.
DEFAULT_TICK_SECONDS = 10.0
# Force a reassurance message after this long with no change, so the user knows
# the search is still alive even when nothing new has appeared.
DEFAULT_HEARTBEAT_SECONDS = 60.0


class _Snapshotter(Protocol):
    def snapshot(self) -> list[dict]: ...


def short_camera(name: str) -> str:
    """Trim ``camera-192.168.1.8`` to ``.8`` for terse, readable updates."""
    host = name.removeprefix("camera-")
    if host.count(".") == 3:
        return "." + host.rsplit(".", 1)[1]
    return name


def currently_visible(rows: list[dict], fresh_seconds: float) -> dict[str, list[str]]:
    """Map each freshly-seen label to the cameras seeing it.

    ``rows`` are :meth:`lib.objects.ObjectRegistry.snapshot` entries (each has a
    ``camera``, a ``label`` and ``since`` = seconds since last seen). Only rows
    within ``fresh_seconds`` count as visible *now*. Cameras are de-duplicated
    and ordered for a stable message.
    """
    visible: dict[str, set[str]] = {}
    for row in rows:
        since = row.get("since")
        if since is None or since > fresh_seconds:
            continue
        visible.setdefault(str(row["label"]).lower(), set()).add(str(row["camera"]))
    return {label: sorted(cameras) for label, cameras in visible.items()}


def format_visible(visible: dict[str, list[str]]) -> str:
    """Render the visible map as ``percy (.8), matcha (.42, .44)``."""
    if not visible:
        return "no birds in view"
    parts = []
    for label in sorted(visible):
        cams = ", ".join(short_camera(camera) for camera in visible[label])
        parts.append(f"{label} ({cams})")
    return "; ".join(parts)


def format_found_message(target: str, cameras: list[str]) -> str:
    where = ", ".join(short_camera(camera) for camera in cameras)
    return f"🔎 Found {target}! On camera {where}."


def format_progress_message(target: str, visible: dict[str, list[str]]) -> str:
    return f"Still looking for {target}. Right now I can see: {format_visible(visible)}."


def format_not_found_message(target: str, elapsed: float, ever_seen: dict[str, list[str]]) -> str:
    minutes = int(round(elapsed / 60))
    base = (
        f"😕 Couldn't find {target} after {minutes} min. "
        "They may be out of frame, behind something, or resting."
    )
    if ever_seen:
        base += f" While searching I did see: {format_visible(ever_seen)}."
    return base


@dataclass
class FindOutcome:
    target: str
    found: bool
    cameras: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class BirdFinder:
    """Runs at most one search at a time, on a background thread.

    A single-flight guard keeps a second ``/find`` from piling on while one is
    already running; the caller gets told a search is in progress instead.
    """

    def __init__(
        self,
        registry: _Snapshotter,
        known_labels: Callable[[], list[str]],
        *,
        notify: Callable[[int, str], None],
        grab_frame: "Callable[[str], bytes | None] | None" = None,
        send_album: "Callable[[int, Sequence[tuple[bytes, str | None]]], None] | None" = None,
        make_patrol: "Callable[[], PtzPatrol | None] | None" = None,
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
        # Grabs a camera's latest frame as JPEG bytes (proof photos on a hit),
        # and sends an album to a chat. Both optional — without them /find still
        # reports in text. Reuses the notifier's rate-limited album sender.
        self._grab_frame = grab_frame
        self._send_album = send_album
        # Builds a fresh patrol per search (reflecting the cameras live right
        # now), or None to search without moving anything. Best-effort.
        self._make_patrol = make_patrol
        self._fresh_seconds = fresh_seconds
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds
        self._tick_seconds = tick_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._clock = clock
        self._active_lock = threading.Lock()
        self._active_target: str | None = None

    # -- validation --------------------------------------------------------

    def findable_labels(self) -> list[str]:
        """Roster labels the live model can actually detect (no generic classes)."""
        return self._known_labels()

    def normalise_target(self, target: str) -> str | None:
        """Return the canonical label for ``target`` if findable, else ``None``."""
        wanted = target.strip().lower()
        if not wanted:
            return None
        for label in self._known_labels():
            if label.lower() == wanted:
                return label.lower()
        return None

    # -- entry point -------------------------------------------------------

    def start(self, chat_id: int, target: str, stop_event: threading.Event) -> str:
        """Validate and launch a search; return the immediate ack/refusal text."""
        labels = ", ".join(self.findable_labels()) or "(model has no labels)"
        if not target.strip():
            return f"Usage: /find <bird>. I can look for: {labels}."
        canonical = self.normalise_target(target)
        if canonical is None:
            return (
                f"I don't know a bird called \"{target.strip()}\". "
                f"I can look for: {labels}."
            )

        with self._active_lock:
            if self._active_target is not None:
                return f"Already searching for {self._active_target}. One hunt at a time."
            self._active_target = canonical

        thread = threading.Thread(
            target=self._run_guarded,
            args=(chat_id, canonical, stop_event),
            name=f"find-{canonical}",
            daemon=True,
        )
        thread.start()
        minutes = int(round(self._timeout_seconds / 60))
        return (
            f"🔭 On it — searching all cameras for {canonical} (up to {minutes} min). "
            "I'll report what I can see and ping you the moment I spot them."
        )

    def _run_guarded(self, chat_id: int, target: str, stop_event: threading.Event) -> None:
        try:
            self._run(chat_id, target, stop_event)
        except Exception:
            LOGGER.exception("Find loop failed for target=%s", target)
            try:
                self._notify(chat_id, f"Search for {target} hit an error and stopped.")
            except Exception:
                LOGGER.exception("Failed to send find-error message")
        finally:
            with self._active_lock:
                self._active_target = None

    # -- the search loop ---------------------------------------------------

    def _run(self, chat_id: int, target: str, stop_event: threading.Event) -> FindOutcome:
        # Build the patrol for THIS search so it reflects the cameras live right
        # now; tear it down in the finally so movement always halts.
        patrol = None
        if self._make_patrol is not None:
            try:
                patrol = self._make_patrol()
            except Exception:
                LOGGER.exception("Building PTZ patrol failed; searching without it")
        try:
            return self._search(chat_id, target, stop_event, patrol)
        finally:
            if patrol is not None:
                try:
                    patrol.stop()
                except Exception:
                    LOGGER.exception("PTZ patrol stop failed")

    def _search(self, chat_id, target, stop_event, patrol) -> FindOutcome:
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

        while not stop_event.is_set() and self._clock() < deadline:
            visible = currently_visible(self._registry.snapshot(), self._fresh_seconds)
            # Accumulate everything seen during the hunt for the not-found recap.
            for label, cams in visible.items():
                merged = set(ever_seen.get(label, [])) | set(cams)
                ever_seen[label] = sorted(merged)

            if target in visible:
                cameras = visible[target]
                self._notify(chat_id, format_found_message(target, cameras))
                self._send_found_photos(chat_id, target, cameras)
                return FindOutcome(target, True, cameras, self._clock() - start)

            now = self._clock()
            if now >= next_tick:
                next_tick = now + self._tick_seconds
                keys = frozenset(visible)
                changed = keys != last_visible_keys
                stale = (now - last_message_at) >= self._heartbeat_seconds
                if changed or stale:
                    self._notify(chat_id, format_progress_message(target, visible))
                    last_visible_keys = keys
                    last_message_at = now

            if patrol is not None:
                try:
                    patrol.step()
                except Exception:
                    LOGGER.exception("PTZ patrol step failed")

            stop_event.wait(self._poll_seconds)

        elapsed = self._clock() - start
        self._notify(chat_id, format_not_found_message(target, elapsed, ever_seen))
        return FindOutcome(target, False, [], elapsed)

    def _send_found_photos(self, chat_id: int, target: str, cameras: list[str]) -> None:
        """Send the latest frame of each camera that saw the target, as proof.

        Best-effort: a camera without a current frame is skipped, and any send
        error is swallowed so it never turns a successful find into a failure.
        """
        if self._grab_frame is None or self._send_album is None:
            return
        items: list[tuple[bytes, str | None]] = []
        for camera in cameras:
            try:
                image = self._grab_frame(camera)
            except Exception:
                LOGGER.exception("Grabbing proof frame for %s failed", camera)
                image = None
            if image:
                items.append((image, f"{target} — camera {short_camera(camera)}"))
        if not items:
            return
        try:
            self._send_album(chat_id, items)
        except Exception:
            LOGGER.exception("Sending found photos failed")


class PtzPatrol(Protocol):
    """Best-effort camera-movement driver used while a search runs."""

    def start(self) -> None: ...
    def step(self) -> None: ...
    def stop(self) -> None: ...
