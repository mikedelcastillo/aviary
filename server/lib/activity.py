"""Read the collected-photos tree as a bird activity log.

Every alerted detection the server collects is already written to
``data/server/collect/<bird>/<stem>.jpg`` with a ``.json`` sidecar holding the
bird, camera, time, confidence and bounding box. That tree IS a timestamped
activity log — so the daycare digest (waves of "what the birds are up to") and
the "what did Percy do today?" Q&A both build on the same reader here, with no
extra database.

The disk-scanning and selection are plain functions (unit-tested); the VLM
captioning and LLM summarising are thin wrappers over :mod:`lib.ai`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from lib.ai.chat import strip_thinking
from lib.labels import pretty


LOGGER = logging.getLogger("lib.activity")


@dataclass(frozen=True)
class Sighting:
    """One collected detection: a bird seen on a camera at a time, with its photo."""

    path: Path
    label: str
    camera: str
    collected_at: float  # epoch seconds
    confidence: float
    bbox: tuple[int, int, int, int]
    width: int
    height: int


@dataclass
class _BoxDetection:
    """Minimal duck-typed detection for build_detection_context (used by the
    activity harness to ground a re-caption in a sighting's saved box)."""

    label: str
    bbox_xyxy: tuple[int, int, int, int]


def _parse_sidecar(json_path: Path) -> Sighting | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        detection = data["detection"]
        bbox = detection["bbox_xyxy"]
        image = json_path.with_suffix(".jpg")
        if not image.exists():
            return None
        from datetime import datetime

        collected_at = datetime.fromisoformat(data["collected_at"]).timestamp()
        frame = data.get("frame", {})
        return Sighting(
            path=image,
            label=str(data["object"]).lower(),
            camera=str(data["camera"]["name"]),
            collected_at=collected_at,
            confidence=float(detection.get("confidence", 0.0)),
            bbox=(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]),
            width=int(frame.get("width", 0)),
            height=int(frame.get("height", 0)),
        )
    except Exception:
        LOGGER.debug("Skipping unreadable sidecar %s", json_path, exc_info=True)
        return None


def load_sightings(collect_dir: Path, since_epoch: float, until_epoch: float | None = None) -> list[Sighting]:
    """All collected sightings in ``[since_epoch, until_epoch]``, newest first.

    Scans ``collect_dir/<bird>/*.json`` (skipping the ``snapshots`` folder, which
    holds on-demand /snapshot frames, not detections).
    """
    if not collect_dir.exists():
        return []
    sightings: list[Sighting] = []
    for label_dir in collect_dir.iterdir():
        if not label_dir.is_dir() or label_dir.name == "snapshots":
            continue
        for json_path in label_dir.glob("*.json"):
            sighting = _parse_sidecar(json_path)
            if sighting is None:
                continue
            if sighting.collected_at < since_epoch:
                continue
            if until_epoch is not None and sighting.collected_at > until_epoch:
                continue
            sightings.append(sighting)
    sightings.sort(key=lambda s: s.collected_at, reverse=True)
    return sightings


def select_highlights(sightings: list[Sighting], max_photos: int = 6) -> list[Sighting]:
    """Pick a diverse, high-quality subset: the best shot of each bird first.

    One strongest (highest-confidence) photo per bird gives variety; if there's
    still room, the next-best distinct shots fill in. Newest wins ties so a
    digest leans recent.
    """
    by_bird: dict[str, list[Sighting]] = {}
    for sighting in sightings:
        by_bird.setdefault(sighting.label, []).append(sighting)

    chosen: list[Sighting] = []
    for shots in by_bird.values():
        chosen.append(max(shots, key=lambda s: (s.confidence, s.collected_at)))
    chosen.sort(key=lambda s: (s.confidence, s.collected_at), reverse=True)
    if len(chosen) >= max_photos:
        return chosen[:max_photos]

    # Backfill with the next-best shots not already chosen.
    picked = set(id(s) for s in chosen)
    remaining = sorted(
        (s for s in sightings if id(s) not in picked),
        key=lambda s: (s.confidence, s.collected_at),
        reverse=True,
    )
    chosen.extend(remaining[: max_photos - len(chosen)])
    return chosen


def summarise_counts(sightings: list[Sighting]) -> str:
    """A terse "Percy ×4, Matcha ×2" tally of who was seen, most-seen first."""
    counts: dict[str, int] = {}
    for sighting in sightings:
        counts[sighting.label] = counts.get(sighting.label, 0) + 1
    if not counts:
        return "no birds"
    parts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{pretty(label)} ×{count}" for label, count in parts)


# -- VLM / LLM wrappers ------------------------------------------------------


_ACTIVITY_SUMMARY_PROMPT = (
    "You are the aviary caretaker giving an activity update from logged memories. "
    "Write it as bullet points — start each line with '• ' — UP TO 10 bullets. "
    "Cover EVERY bird that appears in the notes: at least one bullet per bird, "
    "saying what it did and who it was with. Refer to each bird by NAME, never its "
    "species. Write NATURALLY: use a pronoun (she/he, per the pronouns given) only "
    "where one naturally fits in a sentence — NEVER write a pronoun in parentheses "
    "or right after a name (no 'Percy (she)', no 'Percy, she,'); the pronouns are "
    "given for your reference, not to be quoted. When a note begins with how long "
    "ago it was (e.g. '2 hours ago', '12 minutes ago'), include that timing in the "
    "bullet naturally (e.g. 'Percy preened on the perch 2 hours ago'). Be warm and "
    "concrete; don't invent anything not in the notes. No preamble or closing line "
    "— just the bullets."
)


_ACTIVITY_QA_PROMPT = (
    "You are the warm caretaker of a home aviary, answering a question about the "
    "pet birds using ONLY the timestamped memory notes provided. Each note line is "
    "'(when) [birds seen]: what they were doing'. Answer the SPECIFIC question "
    "directly in 1-4 sentences. Be concrete about WHEN (e.g. 'around 2pm', 'this "
    "morning', 'about an hour ago') and WHICH birds. For a question about two or "
    "more birds being TOGETHER, only say yes if a note lists them together at the "
    "same time, and say when. For a yes/no question, start with a clear yes or no. "
    "If the notes don't show what's asked, say you didn't catch it / can't tell "
    "from today's memory — never guess. Use each bird's correct pronoun (per the "
    "pronouns given); refer to birds by NAME, never the species. No preamble."
)


def answer_activity_question(
    client,
    model: str,
    question: str,
    notes: list[str],
    pronoun_note: str = "",
    window_phrase: str = "",
    *,
    timeout_seconds: float | None = None,
) -> str:
    """Answer a free-form day-lookback question grounded in the memory notes."""
    if not notes:
        return ""
    body = "\n".join(f"- {line}" for line in notes)
    parts = []
    if pronoun_note:
        parts.append(f"Bird pronouns (use these): {pronoun_note}")
    parts.append(f"Question: {question.strip()}")
    header = f"Memory notes from {window_phrase}" if window_phrase else "Memory notes"
    parts.append(f"{header}:\n{body}")
    reply = client.chat(
        model,
        [
            {"role": "system", "content": _ACTIVITY_QA_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        think=True,
        timeout_seconds=timeout_seconds,
    )
    return strip_thinking(reply)


def summarise_activity(
    client, model: str, notes: list[str], subject: str = "", pronoun_note: str = "", *, timeout_seconds: float | None = None
) -> str:
    """Fold journal memory notes into a <=3-sentence activity report.

    ``pronoun_note`` states each bird's sex (the notes often don't carry it, so
    the model would otherwise default everyone to "he").
    """
    if not notes:
        return ""
    body = "\n".join(f"- {line}" for line in notes)
    ask = f"Summarise {subject}'s activity from these notes:\n{body}" if subject else f"Notes:\n{body}"
    if pronoun_note:
        ask = f"Bird pronouns (use these exactly): {pronoun_note}\n\n{ask}"
    reply = client.chat(
        model,
        [
            {"role": "system", "content": _ACTIVITY_SUMMARY_PROMPT},
            {"role": "user", "content": ask},
        ],
        think=True,
        timeout_seconds=timeout_seconds,
    )
    return strip_thinking(reply)
