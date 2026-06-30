"""Friendly, VLM-derived display names for cameras (instead of IP-based ones).

A camera's stable identity stays ``camera-<host>`` (the key the supervisor,
stats and registry use), but everything the USER sees — /status, /find,
/snapshot — renders a friendly name. After discovery the vision model looks at
each camera's view and proposes a 1-2 word name ("Window Perch", "Food Bowl");
:class:`CameraNamer` keeps the identity→display map and guarantees uniqueness.

Until a camera is named (or if the VLM is off), it falls back to the camera's
IP. The map is cached to disk, so a restart shows the last known names instantly
(keyed by ``camera-<ip>``) while the VLM re-confirms them in the background.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

from lib.ai.vlm import clean_camera_name, describe_image, name_camera_view


LOGGER = logging.getLogger("lib.camera_names")


def fallback_name(camera_name: str) -> str:
    """Until the VLM names it, show the camera's IP — an honest address, not a
    made-up non-descriptive name. ``camera-192.168.1.45`` -> ``192.168.1.45``.

    This only shows transiently: naming retries on every discovery sweep (the
    view may also have moved), so a real descriptive name replaces it.
    """
    return camera_name.removeprefix("camera-")


def unique_name(base: str, used: set[str]) -> str:
    """Return ``base``, or ``base 2`` / ``base 3`` ... if it's already taken."""
    if base and base not in used:
        return base
    index = 2
    while f"{base} {index}" in used:
        index += 1
    return f"{base} {index}"


def _distinct_name(client, model: str, image: bytes, taken: set[str], *, timeout_seconds: float) -> str:
    """Ask the VLM for a name that's DIFFERENT from the ones already taken.

    Used when the first suggestion collides ("Big Cage" vs "Big Cage") — instead
    of a numeric suffix, the model looks again at THIS view and names what makes
    it distinct (a colour, an object, a spot).
    """
    prompt = (
        "This is one of several pet-bird camera views. These names are already "
        f"taken: {', '.join(sorted(taken))}. Give a DIFFERENT, distinctive 1-2 word "
        "name for THIS specific view, based on something unique in it (a colour, an "
        "object, a location). Reply with ONLY the name — no 'view', 'cam' or 'camera'."
    )
    return clean_camera_name(describe_image(client, model, image, prompt, timeout_seconds=timeout_seconds))


class CameraNamer:
    """Thread-safe identity -> friendly-name map with a sensible fallback.

    When given ``cache_path`` the map is persisted to JSON and reloaded on
    start, so a restart shows the last known names immediately (the VLM then
    re-confirms them in the background).
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._names: dict[str, str] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path is not None:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            LOGGER.exception("Could not read camera-name cache %s", self._cache_path)
            return
        if isinstance(data, dict):
            with self._lock:
                self._names.update({str(k): str(v) for k, v in data.items()})
            LOGGER.info("Loaded %d cached camera name(s) from %s", len(data), self._cache_path)

    def _save(self) -> None:
        if self._cache_path is None:
            return
        with self._lock:
            snapshot = dict(self._names)
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._cache_path)
        except Exception:
            LOGGER.exception("Could not write camera-name cache %s", self._cache_path)

    def set(self, camera_name: str, display: str) -> None:
        with self._lock:
            self._names[camera_name] = display
        self._save()

    def claim(self, camera_name: str, base: str) -> str:
        """Atomically assign a UNIQUE display name for ``camera_name`` and store it.

        Uniqueness is computed against the current names (excluding this camera's
        own previous name, so re-naming to the same descriptive name doesn't earn
        a spurious "2") and the store happens under the lock — so two naming
        passes running at once (boot + a /discover re-name) can never hand the
        same name to two different cameras. Returns the assigned display.
        """
        with self._lock:
            used = set(self._names.values())
            used.discard(self._names.get(camera_name))
            display = unique_name(base, used)
            self._names[camera_name] = display
        self._save()
        return display

    def has(self, camera_name: str) -> bool:
        with self._lock:
            return camera_name in self._names

    def display(self, camera_name: str) -> str:
        with self._lock:
            name = self._names.get(camera_name)
        return name or fallback_name(camera_name)

    def used_names(self) -> set[str]:
        with self._lock:
            return set(self._names.values())


def name_cameras(
    namer: CameraNamer,
    camera_names: list[str],
    grab_frame: Callable[[str], bytes | None],
    client,
    model: str,
    *,
    stop_event: threading.Event | None = None,
    timeout_seconds: float = 60.0,
    frame_wait_seconds: float = 2.0,
    frame_attempts: int = 8,
    name_attempts: int = 3,
    force: bool = False,
    is_movable: Callable[[str], bool] | None = None,
) -> None:
    """Name cameras from a current frame (best-effort, blocking).

    Runs on a background thread after discovery. For each camera it waits for a
    frame, then asks the vision model for a descriptive name, retrying with a
    fresh frame if the model returns nothing usable (all banned/empty). A camera
    the model still can't name is left UNNAMED (it shows its IP and is retried
    on the next sweep). With ``force`` it RE-names already-named cameras too —
    used on /discover, since a pan-tilt camera may have moved to a new view; an
    existing name is kept if the re-name fails.

    ``is_movable`` lets force skip re-naming a FIXED, already-named camera: only
    a pan-tilt camera's view changes, so re-running the VLM on fixed ones every
    sweep is wasted GPU/cluster work. Unnamed cameras are always (re)tried.
    """
    skipped_fixed = 0
    for camera_name in camera_names:
        already_named = namer.has(camera_name)
        if already_named and not force:
            continue
        # Force re-name: a fixed camera's view doesn't move, so keep its cached
        # name instead of spending a VLM call to re-derive the same thing.
        if already_named and force and is_movable is not None and not is_movable(camera_name):
            skipped_fixed += 1
            continue
        if stop_event is not None and stop_event.is_set():
            return
        image = None
        for _ in range(frame_attempts):
            image = grab_frame(camera_name)
            if image:
                break
            if stop_event is not None:
                stop_event.wait(frame_wait_seconds)
        if not image:
            continue

        base = ""
        for _ in range(max(1, name_attempts)):
            try:
                base = name_camera_view(client, model, image, timeout_seconds=timeout_seconds)
            except Exception:
                LOGGER.exception("VLM naming failed for %s", camera_name)
                base = ""
            if base:
                break
            # Re-grab a fresh frame and try again — a blurry/empty frame or a
            # banned-word answer shouldn't doom the camera to a placeholder.
            fresh = grab_frame(camera_name)
            if fresh:
                image = fresh
            if stop_event is not None:
                stop_event.wait(1.0)

        if not base:
            # Keep an existing good name if a re-name failed; otherwise leave it
            # unnamed (shows its IP) to be retried on the next sweep.
            if not already_named:
                LOGGER.warning("Could not name %s yet; shows its IP, will retry", camera_name)
            continue

        # Snapshot current names — excluding THIS camera's own — only to DECIDE
        # whether to ask the VLM for a more distinctive name. The actual unique
        # assignment is atomic via claim(), so a concurrent naming pass can't
        # collide even though this snapshot may be stale by the time we claim.
        used = namer.used_names() - {namer.display(camera_name)}
        if base in used:
            try:
                distinct = _distinct_name(client, model, image, used, timeout_seconds=timeout_seconds)
            except Exception:
                LOGGER.exception("VLM disambiguation failed for %s", camera_name)
                distinct = ""
            if distinct and distinct not in used:
                base = distinct
        display = namer.claim(camera_name, base)
        LOGGER.info("Named camera %s -> %s", camera_name, display)

    if skipped_fixed:
        LOGGER.info("Re-name skipped %d fixed camera(s) with a cached name", skipped_fixed)
