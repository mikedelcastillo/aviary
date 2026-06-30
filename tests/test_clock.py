from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lib.clock import PH_TZ, now_ph


def test_philippines_is_utc_plus_8_no_dst() -> None:
    # The Philippines has no DST, so the offset is +8h year-round.
    for month in (1, 6, 12):
        off = datetime(2026, month, 15, 12, 0, tzinfo=PH_TZ).utcoffset()
        assert off == timedelta(hours=8)


def test_now_ph_is_naive_manila_walltime() -> None:
    n = now_ph()
    assert n.tzinfo is None  # naive, to compare with journal times
    expected = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
    assert abs((n - expected).total_seconds()) < 5
