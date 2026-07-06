"""Assemble grounding context for the chat/Q&A path.

The free-text "chat" reply is grounded in the live SYSTEM state (time, day/night,
camera health, who's visible, privacy/auto-find) so "how are things?", "is it
dark yet?", "are the cameras ok?" get real answers instead of blind chit-chat.

``format_system_state`` is pure (takes already-resolved values) so it is unit
testable; the live wiring in ``main`` gathers the values and passes the block via
:func:`build_chat_context`.
"""

from __future__ import annotations

from datetime import datetime


def format_system_state(
    now: datetime,
    *,
    paused: bool = False,
    pause_status: str = "",
    cameras_total: int = 0,
    cameras_healthy: int = 0,
    visible_text: str = "",
    daylight: str = "day",
    autofind_on: bool | None = None,
) -> str:
    """A compact "current state" block for the LLM (pure; values pre-resolved).

    ``daylight`` is "day", "night" (all cameras in IR) or "mixed". ``visible_text``
    is an already-formatted "Percy (Big Cage); Pizza (Pool Table)" or "" when
    nothing is in view. ``autofind_on`` is None when the feature isn't wired.
    """
    lines = [f"- It is {now.strftime('%H:%M')} on {now.strftime('%A')}, Philippine time."]

    if paused:
        # control.status() is already a complete, user-facing sentence; use it
        # directly rather than wrapping it (which double-printed "privacy mode").
        # No daylight/sightings lines while paused — the cameras aren't watching.
        lines.append(
            f"- {pause_status.strip()}"
            if pause_status
            else "- Privacy mode is ON — the cameras are not watching or recording right now."
        )
    elif cameras_total <= 0:
        lines.append("- No cameras are online yet (none discovered).")
    else:
        # Daylight/IR and sightings only make sense when cameras are watching.
        lines.append(f"- {cameras_healthy} of {cameras_total} camera(s) are healthy and live.")
        if daylight == "night":
            lines.append("- Every camera is in night/IR mode — it is dark and the birds are most likely roosting/asleep.")
        elif daylight == "mixed":
            lines.append("- Some cameras are in night/IR (dark) and some in daylight.")
        else:
            lines.append("- The cameras are in daylight.")
        if visible_text:
            lines.append(f"- Seen in the last few seconds: {visible_text}.")
        else:
            lines.append("- No birds are visible on any camera right this second (they may be out of frame or resting).")

    if autofind_on is not None:
        lines.append(f"- Auto-find (auto-search for missing birds) is {'ON' if autofind_on else 'OFF'}.")

    header = (
        "Current aviary state (use it to answer questions about right now; do not "
        "invent anything beyond it — to locate a specific bird live, suggest \"find <bird>\"):"
    )
    return header + "\n" + "\n".join(lines)


def build_chat_context(text: str = "", *, system_state: str | None = None) -> str | None:
    """The grounding block for the chat path — the live system state, or None.

    ``text`` is accepted (callers pass the message) but unused: grounding is now
    purely the live state. ``system_state`` is None for callers without live-state
    plumbing (e.g. the console), which then get no grounding block.
    """
    return system_state or None
