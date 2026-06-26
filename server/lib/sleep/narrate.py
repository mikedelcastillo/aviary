"""Warm, human copy for sleep reports — deterministic, with an optional LLM line.

Every public formatter returns a guaranteed deterministic string from a
:class:`~lib.sleep.model.SleepNight` (or a list of them); :func:`llm_summary` may
dress one up via Ollama but ALWAYS falls back to the deterministic template on
any error, so a report never blocks or crashes the tracker thread. Grounding
(the 10-12h target, the cockatiel night-fright note) comes from :mod:`lib.care`.
"""

from __future__ import annotations

import logging
from datetime import datetime

from lib.care import SLEEP_HOURS
from lib.sleep.model import LIGHT, MOTION, NIGHT_FRIGHT, SleepNight

LOGGER = logging.getLogger("lib.sleep.narrate")

_SPARK = "▁▂▃▄▅▆▇█"

# A night scoring at least this counts toward a "good sleep" streak.
GOOD_SLEEP_SCORE = 80

NO_COVERAGE = "No camera coverage to track sleep right now — I'll start watching once the cameras are reporting. 🌙"
COLD_START_NOTE = "Still learning their routine — the schedule read is a best guess until I've tracked a few nights."


def sleep_streak(recent_newest_first: list[SleepNight], good: int = GOOD_SLEEP_SCORE) -> int:
    """How many of the most-recent consecutive nights scored ``good`` or better.

    ``recent_newest_first`` is the nights newest-first (as :func:`load_recent`
    returns them); the count stops at the first night below the threshold.
    """
    streak = 0
    for night in recent_newest_first:
        if night.score is not None and night.score >= good:
            streak += 1
        else:
            break
    return streak


def _band(score: int) -> str:
    if score >= 90:
        return "a great night 😴"
    if score >= 80:
        return "a good night 🙂"
    if score >= 65:
        return "an okay night"
    if score >= 45:
        return "a rough night 😕"
    return "a poor night 😟"


def _hm(minutes: int | None) -> str:
    minutes = max(0, int(minutes or 0))
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _clock(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    hour12 = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{hour12}:{dt.minute:02d}{ampm}"


def _disturbance_line(night: SleepNight) -> str:
    lights = sum(1 for d in night.disturbances if d.kind == LIGHT)
    frights = [d for d in night.disturbances if d.kind == NIGHT_FRIGHT]
    motions = sum(1 for d in night.disturbances if d.kind == MOTION)
    if not night.disturbances:
        return "Quiet: a calm, dark night — no disturbances."
    parts = []
    if lights:
        parts.append(f"{lights} light{'s' if lights > 1 else ''}")
    if motions:
        parts.append(f"{motions} stir{'s' if motions > 1 else ''}")
    if frights:
        parts.append(f"{len(frights)} possible night-fright{'s' if len(frights) > 1 else ''}")
    return "Disturbances: " + ", ".join(parts) + "."


def _schedule_line(night: SleepNight) -> str:
    if not night.baselined:
        return "Still learning their routine — no usual bedtime to compare to yet."
    consistency = night.components.get("consistency")
    if consistency is None:
        return ""
    if consistency >= 0.9:
        return "On schedule: bedtime and wake-up right around their usual."
    if consistency >= 0.6:
        return "On schedule: a little off their usual bedtime or wake time."
    return "Off schedule: bedtime/wake drifted well off their usual — a steadier routine would help."


def _fright_note(night: SleepNight) -> str | None:
    frights = [d for d in night.disturbances if d.kind == NIGHT_FRIGHT]
    if not frights:
        return None
    when = _clock(frights[0].time)
    return (
        f"⚠️ A possible night-fright around {when} — worth a quick check for any pulled "
        "feathers on Draft or Pizza, and keep that dim red nightlight on for the cockatiels."
    )


def _dark_phrase(night: SleepNight) -> str:
    low, high = SLEEP_HOURS
    h = night.dark_hours
    span = f" ({_clock(night.lights_out)} → {_clock(night.first_light)})" if night.lights_out and night.first_light else ""
    if low <= h <= high:
        tail = f" — right in the {low}–{high}h sweet spot."
    elif h < low:
        tail = f" — a little short of the {low}–{high}h they need."
    else:
        tail = " — plenty of dark."
    return f"Dark: {_hm(night.dark_minutes)}{span}{tail}"


def format_last(night: SleepNight, *, streak: int = 0) -> str:
    """The full last-night report. ``streak`` is the run of good nights so far."""
    lines = [
        f"🌙 Last night ({night.night_of:%a %b %d})",
        f"Sleep score: {night.score}/100 · {_band(night.score or 0)}",
        _dark_phrase(night),
        _disturbance_line(night),
    ]
    schedule = _schedule_line(night)
    if schedule:
        lines.append(schedule)
    if streak >= 2:
        lines.append(f"🔥 {streak} good nights in a row!")
    if night.confidence < 0.7:
        lines.append("(Reading is a best guess — partial camera coverage.)")
    fright = _fright_note(night)
    if fright:
        lines.append(fright)
    return "\n".join(lines)


def format_morning(night: SleepNight) -> str:
    """A single warm morning one-liner fired right after 'Good morning'."""
    base = (
        f"😴 Sleep report — the flock got {_hm(night.dark_minutes)} of dark last night "
        f"({_clock(night.lights_out)}–{_clock(night.first_light)}), score {night.score}/100. "
        f"{_disturbance_line(night)}"
    )
    fright = _fright_note(night)
    return f"{base}\n{fright}" if fright else base


def _spark(scores: list[int]) -> str:
    if not scores:
        return ""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return _SPARK[3] * len(scores)
    return "".join(_SPARK[round((s - lo) / (hi - lo) * (len(_SPARK) - 1))] for s in scores)


def format_week(nights: list[SleepNight]) -> str:
    """A 7-night trend summary (nights oldest-first)."""
    scored = [n for n in nights if n.score is not None]
    if not scored:
        return "No finished nights to summarise yet — give it a night or two. 🌙"
    scores = [n.score for n in scored]
    avg = round(sum(scores) / len(scores))
    best = max(scored, key=lambda n: n.score)
    worst = min(scored, key=lambda n: n.score)
    avg_dark = round(sum(n.dark_minutes or 0 for n in scored) / len(scored))
    frights = sum(1 for n in scored for d in n.disturbances if d.kind == NIGHT_FRIGHT)
    lines = [
        f"🌙 The birds' sleep — last {len(scored)} night{'s' if len(scored) > 1 else ''}",
        f"Avg score {avg}/100 · trend {_spark(scores)}",
        f"• Best: {best.night_of:%a} — {best.score}, a {_hm(best.dark_minutes)} night.",
        f"• Worst: {worst.night_of:%a} — {worst.score}.",
        f"Dark hours: avg {_hm(avg_dark)} (aim {SLEEP_HOURS[0]}–{SLEEP_HOURS[1]}h).",
    ]
    if frights:
        lines.append(f"{frights} possible night-fright{'s' if frights > 1 else ''} this week — keep the dim red nightlight on for Draft and Pizza.")
    return "\n".join(lines)


def format_status_line(in_progress: SleepNight | None, now: datetime) -> str | None:
    """A single '/status' line while a night is open, or None."""
    if in_progress is None or in_progress.lights_out is None:
        return None
    minutes = max(0, int((now - in_progress.lights_out).total_seconds() / 60))
    return f"🌙 Night in progress — dark {_hm(minutes)} so far"


def llm_summary(client, model: str, night: SleepNight, *, care_context: str = "", timeout_seconds: float | None = None) -> str:
    """An optional Ollama-written one-liner, with the deterministic morning line
    as the guaranteed fallback on ANY error."""
    fallback = format_morning(night)
    if client is None:
        return fallback
    try:
        from lib.ai.chat import clean_reply

        facts = (
            f"dark {_hm(night.dark_minutes)} ({_clock(night.lights_out)}–{_clock(night.first_light)}), "
            f"score {night.score}/100, {_disturbance_line(night)}"
        )
        system = (
            "You are the warm caretaker of a home aviary. In ONE or two friendly sentences, "
            "tell the owner how the flock slept from these facts. Don't invent anything. "
            + care_context
        )
        reply = client.chat(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": facts}],
            think=True,
            timeout_seconds=timeout_seconds,
        )
        return clean_reply(reply) or fallback
    except Exception:
        LOGGER.exception("LLM sleep summary failed; using template")
        return fallback
