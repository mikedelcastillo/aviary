"""Assemble grounding context for the chat/Q&A path so it can answer anything.

The free-text "chat" reply used to be state-blind and knowledge-free, so it
could only chit-chat. This module gives it two kinds of grounding:

  * the live SYSTEM state (time, day/night, camera health, who's visible,
    privacy/auto-find) — so "how are things?", "is it dark yet?", "are the
    cameras ok?" get real answers, and
  * the relevant CARE knowledge (:mod:`lib.care`) — so "can they eat avocado?",
    "how long should they sleep?" are answered accurately.

``format_system_state`` is pure (takes already-resolved values) so it is unit
testable; the live wiring in ``main`` gathers the values and joins the two
blocks via :func:`build_chat_context`.
"""

from __future__ import annotations

from datetime import datetime

from lib.care import care_context


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

    if daylight == "night":
        lines.append("- Every camera is in night/IR mode — it is dark and the birds are most likely roosting/asleep.")
    elif daylight == "mixed":
        lines.append("- Some cameras are in night/IR (dark) and some in daylight.")
    else:
        lines.append("- The cameras are in daylight.")

    if paused:
        status = pause_status or "Privacy mode is on."
        lines.append(f"- PRIVACY MODE IS ON — cameras are not watching or recording right now ({status.strip()}).")
    elif cameras_total <= 0:
        lines.append("- No cameras are online yet (none discovered).")
    else:
        lines.append(f"- {cameras_healthy} of {cameras_total} camera(s) are healthy and live.")

    if paused:
        pass  # nothing is visible while paused; don't claim sightings
    elif visible_text:
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


def build_chat_context(
    text: str,
    *,
    system_state: str | None = None,
    member_species: dict[str, str] | None = None,
) -> str | None:
    """Join the live system-state block with the care-knowledge block, or None.

    Either block may be absent: ``system_state`` is None for callers without
    live state (e.g. the console), and the care block is None for non-care
    messages. Returns None only when both are absent.
    """
    parts = []
    if system_state:
        parts.append(system_state)
    care = care_context(text, member_species=member_species)
    if care:
        parts.append(care)
    return "\n\n".join(parts) if parts else None
