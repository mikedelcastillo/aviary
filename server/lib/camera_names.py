"""Friendly, VLM-derived display names for cameras (instead of IP-based ones).

A camera's stable identity stays ``camera-<host>`` (the key the supervisor,
stats and registry use), but everything the USER sees — /status, /find,
/snapshot — renders a friendly name. After discovery the vision model looks at
each camera's view and proposes a 1-2 word name ("Window Perch", "Food Bowl");
:class:`CameraNamer` keeps the identity→display map and guarantees uniqueness.

Until a camera is named (or if the VLM is off), it falls back to ``Cam 8`` — the
host's last octet, NOT the bare ``.8`` that read like a stray decimal.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from lib.ai.vlm import name_camera_view


LOGGER = logging.getLogger("lib.camera_names")


def fallback_name(camera_name: str) -> str:
    """A readable placeholder name from the camera identity: camera-x.y.z.8 -> Cam 8."""
    host = camera_name.removeprefix("camera-")
    if host.count(".") == 3:
        return f"Cam {host.rsplit('.', 1)[1]}"
    return camera_name


def unique_name(base: str, used: set[str]) -> str:
    """Return ``base``, or ``base 2`` / ``base 3`` ... if it's already taken."""
    if base and base not in used:
        return base
    index = 2
    while f"{base} {index}" in used:
        index += 1
    return f"{base} {index}"


class CameraNamer:
    """Thread-safe identity -> friendly-name map with a sensible fallback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: dict[str, str] = {}

    def set(self, camera_name: str, display: str) -> None:
        with self._lock:
            self._names[camera_name] = display

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
) -> None:
    """Name any not-yet-named cameras from a current frame (best-effort, blocking).

    Intended to run on a background thread after discovery. For each unnamed
    camera it waits briefly for a frame to arrive, asks the vision model for a
    name, de-duplicates it, and records it. A camera that never yields a frame or
    whose VLM call fails simply keeps its fallback name.
    """
    for camera_name in camera_names:
        if namer.has(camera_name):
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
        try:
            base = name_camera_view(client, model, image, timeout_seconds=timeout_seconds)
        except Exception:
            LOGGER.exception("VLM naming failed for %s", camera_name)
            base = ""
        display = unique_name(base or fallback_name(camera_name), namer.used_names())
        namer.set(camera_name, display)
        LOGGER.info("Named camera %s -> %s", camera_name, display)
