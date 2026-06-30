"""The one stateful/threaded piece of the sleep tracker.

It turns the live IR signal + a motion poll into the ``Ir`` / ``Motion`` / ``Tick``
events the pure :mod:`lib.sleep.detect` reducer consumes, owns persistence (the
monthly JSONL + the in-progress ``_current.json`` sidecar), scores a night the
moment it finalizes (:mod:`lib.sleep.score`), and fires the optional morning
report. The detection/scoring brains are pure; this shell is deliberately thin.

``_tick(now)`` and ``on_ir`` are injectable/callable directly, so the whole thing
unit-tests with a fake clock + fake all_ir/movement and zero real threads —
exactly like :class:`lib.care_scheduler.CareScheduler`.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable

from lib.clock import now_ph
from lib.sleep.detect import Ir, Motion, NightState, Tick, step
from lib.sleep.model import (
    LIGHT,
    SleepNight,
    append_night,
    clear_current,
    load_current,
    load_recent,
    save_current,
)
from lib.sleep.narrate import format_morning, llm_summary
from lib.sleep.score import rolling_baseline, score_night

LOGGER = logging.getLogger("lib.sleep.tracker")

# Prior finalized nights folded into the rolling consistency baseline.
BASELINE_NIGHTS = 7


class SleepTracker:
    def __init__(
        self,
        notify_all: Callable[[str], None],
        stop_event: threading.Event,
        *,
        all_ir: Callable[[], bool],
        camera_count: Callable[[], int],
        movement: Callable[[], float],
        sleep_dir,
        now: Callable[[], datetime] = now_ph,
        poll_seconds: float = 60.0,
        morning_report: bool = False,
        client=None,
        llm_model: str = "",
        care_context: str = "",
    ) -> None:
        self._notify_all = notify_all
        self._stop = stop_event
        self._all_ir = all_ir
        self._camera_count = camera_count
        self._movement = movement
        self._dir = sleep_dir
        self._now = now
        self._poll = poll_seconds
        self._morning_report = morning_report
        self._client = client
        self._llm_model = llm_model
        self._care_context = care_context

        self._lock = threading.Lock()
        self._state = NightState()
        self._last_finalized: SleepNight | None = None
        self._restore()

    # -- restore on boot ---------------------------------------------------

    def _restore(self) -> None:
        night = load_current(self._dir)
        if night is None or night.finalized:
            return
        # Resume the open night; rebuild the re-lit accumulator from its recorded
        # light disturbances. Mark partial — the restart gap loses some fidelity.
        night.partial_coverage = True
        self._state.night = night
        self._state._relit_minutes = sum(d.minutes for d in night.disturbances if d.kind == LIGHT)
        self._state._last_cameras = night.camera_count_at_dark
        # If the room is LIT at resume (cameras reporting, not all-IR), anchor the
        # open lit interval to now so it's excluded from dark when it closes. The
        # pre-restart portion is unavoidably lost; with no cameras yet we make no
        # assumption and let the first live IR event settle the state.
        if self._camera_count() > 0 and not self._all_ir():
            self._state._leave_since = self._now()
            self._state._last_leave_at = self._state._leave_since
        LOGGER.info("Resumed in-progress sleep night from %s", self._dir)

    # -- public ------------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._run, name="sleep-tracker", daemon=True).start()

    def on_ir(self, camera: str, is_ir: bool) -> None:
        """ir_state listener: a per-camera IR transition -> one Ir event."""
        now = self._now()
        cameras = self._camera_count()
        dark = self._all_ir() and cameras > 0
        with self._lock:
            step(self._state, Ir(now, dark, cameras))
            self._after_step()

    def last_finalized(self) -> SleepNight | None:
        with self._lock:
            if self._last_finalized is not None:
                return self._last_finalized
        recent = load_recent(self._dir, 1)
        return recent[0] if recent else None

    def recent(self, count: int) -> list[SleepNight]:
        return load_recent(self._dir, count)

    def in_progress(self) -> SleepNight | None:
        with self._lock:
            return self._state.night

    # -- loop --------------------------------------------------------------

    def _run(self) -> None:
        LOGGER.info("Sleep tracker started (poll %.0fs, morning report %s)", self._poll, self._morning_report)
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                break
            try:
                self._tick(self._now())
            except Exception:
                LOGGER.exception("Sleep tracker tick failed")

    def _tick(self, now: datetime) -> None:
        with self._lock:
            if self._state.night is not None:
                step(self._state, Motion(now, self._movement()))
                self._after_step()
            # Always advance the clock so a dark-confirm, a wake, or a force-
            # finalize fires even with no IR transition in between.
            step(self._state, Tick(now))
            self._after_step()

    # -- persistence side-effects (lock held) ------------------------------

    def _after_step(self) -> None:
        finalized = self._state.just_finalized
        if finalized is not None:
            self._finalize(finalized)
        elif self._state.night is not None:
            try:
                save_current(self._dir, self._state.night)
            except Exception:
                LOGGER.exception("Saving in-progress sleep night failed")

    def _finalize(self, night: SleepNight) -> None:
        baseline = rolling_baseline(load_recent(self._dir, BASELINE_NIGHTS))
        night.baselined = baseline is not None
        night.score, night.components, night.confidence = score_night(night, baseline)
        night.summary = format_morning(night)
        try:
            append_night(self._dir, night)
            clear_current(self._dir)
        except Exception:
            LOGGER.exception("Persisting finalized sleep night failed")
        self._last_finalized = night
        LOGGER.info("Sleep night finalized: score=%s dark=%smin", night.score, night.dark_minutes)
        if self._morning_report:
            try:
                self._notify_all(llm_summary(self._client, self._llm_model, night, care_context=self._care_context))
            except Exception:
                LOGGER.exception("Sleep morning report failed")
