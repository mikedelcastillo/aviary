"""Automatic searches for birds that have gone missing (``/autofind``).

Polls the sighting registry; when a bird that was being seen hasn't appeared for
over ``unseen_seconds`` (default 10 min) it launches ONE batched ``/find`` for
all the overdue birds at once. Searching is gated on daylight: if ANY camera is
in night/IR mode the birds can't be told apart (and are likely roosting), so no
search runs. The feature also auto-toggles — enabling when every camera is back
in daylight and disabling when all cameras go IR — announcing each change.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from lib.find import bird_last_seen
from lib.labels import pretty


LOGGER = logging.getLogger("lib.autofind")


def _phrase(birds: list[str]) -> str:
    names = [pretty(b) for b in birds]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


class AutoFinder:
    def __init__(
        self,
        registry,
        finder,
        control,
        notifier,
        stop_event: threading.Event,
        ir_cameras: Callable[[], set[str]],
        camera_count: Callable[[], int],
        known_birds: list[str],
        *,
        unseen_seconds: float = 600.0,
        poll_seconds: float = 60.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._finder = finder
        self._control = control
        self._notifier = notifier
        self._stop = stop_event
        self._ir_cameras = ir_cameras
        self._camera_count = camera_count
        self._known_birds = [b.lower() for b in known_birds]
        self._unseen = unseen_seconds
        self._poll = poll_seconds
        self._now = now
        self._lock = threading.Lock()
        self._enabled = False
        self._manual_override = False  # the user set it explicitly this day-phase
        self._alerted: set[str] = set()
        self._last_condition: str | None = None  # "clear" | "all_ir" | "mixed"

    # -- public ------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, on: bool) -> str:
        with self._lock:
            self._enabled = on
            self._manual_override = True
        state = "on" if on else "off"
        LOGGER.info("Auto-find manually turned %s", state)
        return f"🔎 Auto-find is now {state.upper()}. I'll search for birds missing over {int(self._unseen // 60)} min when the cameras are in daylight."

    def status(self) -> str:
        minutes = int(self._unseen // 60)
        on = self.enabled
        return (
            f"🔎 Auto-find is {'ON' if on else 'OFF'} — "
            f"searches for any bird missing over {minutes} min (paused while any camera is in night/IR). "
            f"Use /autofind enable or /autofind disable."
        )

    def start(self) -> None:
        threading.Thread(target=self._run, name="auto-finder", daemon=True).start()

    # -- loop --------------------------------------------------------------

    def _run(self) -> None:
        LOGGER.info("Auto-finder started (poll %.0fs, missing after %.0fs)", self._poll, self._unseen)
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                break
            try:
                self._tick()
            except Exception:
                LOGGER.exception("Auto-finder tick failed")

    def _tick(self) -> None:
        ir = self._ir_cameras()
        total = self._camera_count()
        self._auto_toggle(len(ir), total)

        if not self.enabled:
            return
        if ir:
            return  # any camera in IR -> can't tell birds apart; don't search
        if self._control.is_paused():
            return
        if self._finder.is_searching():
            return

        overdue = self._overdue()
        if not overdue:
            return
        with self._lock:
            self._alerted.update(overdue)
        minutes = int(self._unseen // 60)
        # The search's own progress/result go to one chat (finder.start targets a
        # single chat_id), so the kickoff notice goes to that SAME chat — not a
        # broadcast that promises every user a result only one of them receives.
        chat_id = self._notifier.user_ids[0] if getattr(self._notifier, "user_ids", None) else 0
        self._notifier.send_text(
            chat_id,
            f"🔍 Auto-find: haven't seen {_phrase(overdue)} in over {minutes} min — searching all cameras…",
        )
        self._finder.start(chat_id, " and ".join(overdue), self._stop)

    def _overdue(self) -> list[str]:
        last = bird_last_seen(self._registry.snapshot())
        overdue: list[str] = []
        came_back: set[str] = set()
        for bird in self._known_birds:
            info = last.get(bird)
            if info is None:
                continue  # never seen since boot — no baseline to call it "missing"
            since = info[0]
            if since <= self._unseen:
                came_back.add(bird)
            elif bird not in self._alerted:
                overdue.append(bird)
        if came_back:
            with self._lock:
                self._alerted.difference_update(came_back)  # a returned bird can alert again
        return overdue

    def _auto_toggle(self, ir_count: int, total: int) -> None:
        if total <= 0:
            return
        condition = "clear" if ir_count == 0 else "all_ir" if ir_count >= total else "mixed"
        with self._lock:
            first = self._last_condition is None
            if condition == self._last_condition:
                return
            self._last_condition = condition
            # A real day/night transition clears any manual override.
            if not first:
                self._manual_override = False
            announce = None
            if condition == "clear":
                # New day: forget yesterday's alerts so a bird that's STILL
                # missing this morning triggers a fresh search instead of staying
                # suppressed from last night.
                self._alerted.clear()
            if condition == "clear" and not self._enabled and not (first and self._manual_override):
                self._enabled = True
                announce = None if first else "🟢 Auto-find re-enabled — all cameras are back in daylight."
            elif condition == "all_ir" and self._enabled:
                self._enabled = False
                announce = None if first else "🌙 Auto-find paused — all cameras switched to night/IR."
        if announce:
            self._notifier.broadcast_text(announce)
