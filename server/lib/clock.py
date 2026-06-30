"""Wall-clock time for user-facing reports — always Philippine time.

Every timestamp the user sees (activity bullets, caretaker reports, the journal,
the dashboard clock) goes through :func:`now_ph` so it reads in Asia/Manila no
matter what timezone the host is set to. Returned naive (no tzinfo) so it stays
directly comparable with the journal's parsed ``HH:MM`` times.

This is deliberately NOT used for storage/protocol timestamps that must be UTC
(ONVIF WS-Security in :mod:`lib.ptz`, the collect/snapshot sidecar times).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

PH_TZ = ZoneInfo("Asia/Manila")


def now_ph() -> datetime:
    """Current Asia/Manila wall-clock time, tz-naive."""
    return datetime.now(PH_TZ).replace(tzinfo=None)
