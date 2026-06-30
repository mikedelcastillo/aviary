"""Human-readable duration formatting, shared so every surface agrees.

A single formatter avoids the drift that comes from re-implementing the same
h/m/s math in more than one place (e.g. one copy growing a days bucket the other
lacks). Used by ``/detections`` totals and the Telegram status helpers.
"""

from __future__ import annotations


def format_duration(seconds: float | None) -> str:
    """Compact ``Nd Nh`` / ``Nh Nm`` / ``Nm Ns`` / ``Ns`` (``never`` for None)."""
    if seconds is None:
        return "never"
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
