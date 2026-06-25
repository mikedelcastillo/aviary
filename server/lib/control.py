"""Runtime privacy/pause control shared across the camera workers.

A single :class:`RuntimeControl` is created in ``main`` and handed to every
camera monitor thread (and to the snapshot command). When the user pauses —
``/pause`` / ``/stop`` on Telegram, or a natural-language "stop the cams" — the
capture loops stop *consuming the RTSP streams entirely*: no decoding, no
inference, no recording, no alerts. That is the strongest possible privacy
guarantee — the bytes are never pulled off the camera — rather than merely
muting notifications while still watching.

Auto-resume is deadline-based and *lazy*: a paused-with-duration state simply
records when it should lapse, and :meth:`is_paused` clears itself once the
deadline passes. The capture loops already poll ``is_paused`` while idling, so a
timed pause un-pauses on its own within one poll interval with no extra timer
thread to own or shut down.
"""

from __future__ import annotations

import re
import threading
import time


def _format_remaining(seconds: float) -> str:
    """Render a remaining-time countdown like ``2m 5s`` / ``1h 3m`` / ``8s``."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3_600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# Unit -> seconds. Longest spellings first so the regex below prefers "minutes"
# over a bare "m" inside it. Covers the casual inputs the user asked for: "1m",
# "1 min", "1 minute", "1h", "2 hours", "30s", "1d".
_UNIT_SECONDS: dict[str, int] = {
    "d": 86_400,
    "day": 86_400,
    "days": 86_400,
    "h": 3_600,
    "hr": 3_600,
    "hrs": 3_600,
    "hour": 3_600,
    "hours": 3_600,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
}

# One "<number><unit>" clause, unit optional. Whitespace between number and unit
# is allowed ("1 min") and several clauses may run together ("1h30m").
_DURATION_CLAUSE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]*)")


def parse_duration(text: str | None) -> float | None:
    """Parse a casual duration string into seconds, or ``None`` for indefinite.

    Accepts the loose forms a person actually types: ``"1m"``, ``"1 min"``,
    ``"1 minute"``, ``"1h"``, ``"2 hours"``, ``"30s"``, ``"1h30m"``. A bare
    number with no unit is read as MINUTES — "pause 5" overwhelmingly means five
    minutes, not five seconds. Empty / unparseable input returns ``None`` so the
    caller treats it as an indefinite pause.
    """
    if not text:
        return None
    text = text.strip().lower()
    if not text:
        return None

    total = 0.0
    matched = False
    for value, unit in _DURATION_CLAUSE.findall(text):
        amount = float(value)
        if unit == "":
            # A bare number defaults to minutes (the common "pause 5" case).
            total += amount * 60
            matched = True
        elif unit in _UNIT_SECONDS:
            total += amount * _UNIT_SECONDS[unit]
            matched = True
        else:
            # An unrecognised unit token (e.g. "1x") is ignored rather than
            # guessed at; if nothing else matched we fall through to indefinite.
            continue
    if not matched or total <= 0:
        return None
    return total


class RuntimeControl:
    """Thread-safe pause/resume (privacy) state for stream consumption.

    Readers (the camera loops) call :meth:`is_paused` on the hot path; writers
    (the Telegram command thread) call :meth:`pause` / :meth:`resume`. A single
    short-held lock guards the two fields, so neither side ever blocks the other
    for more than a couple of field reads.
    """

    def __init__(self, *, clock=time.monotonic) -> None:
        self._lock = threading.Lock()
        self._paused = False
        # Wall-deadline (in ``clock`` units) at which a timed pause lapses, or
        # ``None`` for an indefinite pause. Only meaningful while ``_paused``.
        self._resume_at: float | None = None
        self._clock = clock

    def pause(self, duration_seconds: float | None = None) -> str:
        """Enter privacy mode, optionally auto-resuming after ``duration_seconds``.

        ``None`` (or a non-positive duration) pauses indefinitely. Returns a
        human-readable status line describing the new state.
        """
        with self._lock:
            self._paused = True
            if duration_seconds is not None and duration_seconds > 0:
                self._resume_at = self._clock() + duration_seconds
            else:
                self._resume_at = None
        return self.status()

    def resume(self) -> str:
        """Leave privacy mode and resume stream consumption."""
        with self._lock:
            was_paused = self._paused
            self._paused = False
            self._resume_at = None
        return "Resumed — cameras are live again." if was_paused else "Already live."

    def is_paused(self) -> bool:
        """True while privacy mode is active.

        Self-expiring: once a timed pause's deadline passes this flips the state
        back to running and returns False, so the capture loops resume on their
        next poll without any separate timer.
        """
        with self._lock:
            if not self._paused:
                return False
            if self._resume_at is not None and self._clock() >= self._resume_at:
                self._paused = False
                self._resume_at = None
                return False
            return True

    def remaining_seconds(self) -> float | None:
        """Seconds until an active timed pause lapses; ``None`` if indefinite/live."""
        with self._lock:
            if not self._paused or self._resume_at is None:
                return None
            return max(0.0, self._resume_at - self._clock())

    def status(self) -> str:
        """One-line human status: live, or paused with remaining time."""
        with self._lock:
            paused = self._paused
            resume_at = self._resume_at
            now = self._clock()
        if not paused:
            return "Cameras are live."
        if resume_at is None:
            return "Privacy mode on — cameras paused indefinitely. Send /play to resume."
        remaining = max(0.0, resume_at - now)
        return (
            f"Privacy mode on — cameras paused, {_format_remaining(remaining)} "
            "remaining (or /play to resume now)."
        )
