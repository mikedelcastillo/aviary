"""Shared per-camera IR (night-mode) state, updated from the camera loop.

The camera monitor already decodes every frame, so it stamps each camera's IR
flag here once per inference. The rest of the server — /status, auto-find, the
activity feed — then reads a cheap cached value instead of re-decoding frames.
Listeners fire on a per-camera day↔night transition (and every transition is
logged), which is what lets features react to "a light came on" without polling.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

LOGGER = logging.getLogger("lib.ir")

Listener = Callable[[str, bool], None]


class IRState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ir: dict[str, bool] = {}
        # Cameras whose flag is FROZEN: updates are dropped while held. Used
        # while a spotlight is forced on (lib.tapo) — the lamp makes night
        # frames full-colour, and without the freeze that would fake a day
        # transition to every consumer (sleep tracker, auto-find, caretaker).
        self._held: set[str] = set()
        self._listeners: list[Listener] = []

    def add_listener(self, listener: Listener) -> None:
        """Register ``listener(camera, is_ir)``, called on each IR transition."""
        with self._lock:
            self._listeners.append(listener)

    def hold(self, camera: str) -> None:
        """Freeze ``camera``'s flag at its current value; updates are dropped."""
        with self._lock:
            self._held.add(camera)

    def release(self, camera: str) -> None:
        """Unfreeze ``camera``; the next decoded frame re-stamps its flag."""
        with self._lock:
            self._held.discard(camera)

    def update(self, camera: str, is_ir: bool) -> None:
        """Record a camera's current IR flag; fire listeners on a change."""
        with self._lock:
            if camera in self._held:
                return
            previous = self._ir.get(camera)
            self._ir[camera] = is_ir
            if previous == is_ir:
                return
            listeners = list(self._listeners)
        LOGGER.info("Camera %s switched to %s", camera, "night/IR" if is_ir else "daylight")
        for listener in listeners:
            try:
                listener(camera, is_ir)
            except Exception:
                LOGGER.exception("IR listener failed for %s", camera)

    def forget(self, camera: str) -> None:
        """Drop a camera (e.g. when its monitor stops) so it stops counting."""
        with self._lock:
            self._ir.pop(camera, None)

    def is_ir(self, camera: str) -> bool:
        with self._lock:
            return self._ir.get(camera, False)

    def ir_cameras(self) -> set[str]:
        with self._lock:
            return {camera for camera, on in self._ir.items() if on}

    def known(self) -> set[str]:
        """Cameras that have reported at least one frame."""
        with self._lock:
            return set(self._ir)

    def all_ir(self) -> bool:
        """True only when every reporting camera is in night/IR (none if empty)."""
        with self._lock:
            return bool(self._ir) and all(self._ir.values())
