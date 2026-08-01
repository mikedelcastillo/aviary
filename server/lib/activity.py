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
import re
from dataclasses import dataclass
from pathlib import Path

from lib.ai.chat import clean_reply, collect_stream
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


# Written for an INSTRUCT model (gemma3): short, bulleted rules keep it terse,
# on-subject, and out of markdown. The old qwen3-era wall-of-text made gemma3
# ramble in **headers** and answer about the wrong bird.
_ACTIVITY_SUMMARY_PROMPT = (
    "You are the caretaker of a home aviary giving a short activity update from logged "
    "memory notes. Each note is '(when) [birds seen]: what they were doing'.\n\n"
    "Rules:\n"
    "- Output 2-4 bullet lines, each starting with '• ' and under 16 words. Nothing else "
    "— no intro, no heading, no closing line.\n"
    "- Cover the requested bird first; mention another bird only when notable. Say what "
    "happened and roughly when (e.g. '• Percy preened on the perch around noon').\n"
    "- Call each bird by NAME with its correct pronoun (per the pronoun note). NEVER use a "
    "species or breed word (no 'cockatiel', 'parakeet', 'the white bird').\n"
    "- Plain text only: NO markdown, no **bold**, no headers. Be concrete; invent nothing "
    "that is not in the notes."
)


_ACTIVITY_QA_PROMPT = (
    "You are the caretaker of a home aviary, answering ONE question about the pet birds "
    "from timestamped memory notes plus an exact COUNTS summary (tallies of sightings, "
    "activities, and together/apart moments — trust these numbers over the prose).\n\n"
    "Rules:\n"
    "- Answer about the bird(s) NAMED in the question — never drift to a different bird.\n"
    "- For a BROAD question ('what did X do today/this week', 'how is X'), summarise the "
    "OVERALL pattern from the activity counts — e.g. 'Percy spent the week mostly resting "
    "and socialising, preening often, with some play' — not one single moment.\n"
    "- For 'was X with Y' / 'did X spend time with Y': answer YES if they were seen "
    "together even ONCE (say roughly when and how often); answer NO only when the together "
    "count is zero. NEVER say 'no' and then describe them being together.\n"
    "- For a WHOLE-FLOCK question, use the per-bird tallies: 'who was most/least active' "
    "or 'who ate the most' → name the specific top/bottom bird with the number; 'any bird "
    "that didn't eat/play?' → name the birds with NO such activity recorded, but add that "
    "you may simply not have caught it; 'any bird not doing well?' → name any bird with a "
    "health concern and gently suggest an avian vet, or reassure that they all look fine if "
    "none is flagged.\n"
    "- Start with 'Yes' or 'No' ONLY for a yes/no question. An open question ('what did "
    "X do', 'who was most active') just describes — do NOT begin with 'Yes' or 'No'.\n"
    "- Keep it to 2-3 short sentences of plain prose; be concrete about WHEN ('around 2pm', "
    "'this morning', 'early in the week').\n"
    "- Speak naturally, as if you watched them yourself. NEVER quote the data: do not write "
    "'counts', 'facts', 'notes', 'window', 'observations', or a raw '(2 hours ago)' stamp.\n"
    "- Call each bird by NAME with its correct pronoun (per the pronoun note). NEVER use a "
    "species or breed word (no 'cockatiel', 'parakeet', 'the white bird').\n"
    "- Plain text ONLY: no markdown, no **bold**, no headers, no bullet lists.\n"
    "- If the data doesn't cover what was asked, say you didn't catch it from the memory — "
    "never guess or invent."
)


# Open questions ("what did they do", "who was most active") never take a Yes/No
# answer, but the recall model sometimes prepends a stray "Yes,"/"No," anyway. We
# strip it deterministically rather than trusting the prompt rule alone.
_OPEN_Q_STARTS = ("what", "who", "when", "where", "why", "how", "which", "describe", "tell", "show", "list")
_LEADING_YESNO = re.compile(r"^\s*(yes|yeah|yep|no|nope)\b[\s,.:;!—-]*", re.IGNORECASE)


def _strip_stray_yesno(reply: str, question: str) -> str:
    """Drop a leading "Yes"/"No" from the answer to an OPEN question (a yes/no
    question keeps it — the answer legitimately starts there)."""
    q = question.strip().lstrip("\"'“‘ ").lower()
    if not any(q.startswith(w) for w in _OPEN_Q_STARTS):
        return reply
    m = _LEADING_YESNO.match(reply)
    if not m:
        return reply
    rest = reply[m.end():]
    if not rest:
        return reply
    return rest[0].upper() + rest[1:]


def answer_activity_question(
    client,
    model: str,
    question: str,
    notes: list[str],
    pronoun_note: str = "",
    window_phrase: str = "",
    *,
    facts: str = "",
    timeout_seconds: float | None = None,
    on_partial=None,
    cancelled=None,
) -> str:
    """Answer a free-form day-lookback question grounded in the memory notes."""
    if not notes:
        return ""
    body = "\n".join(f"- {line}" for line in notes)
    parts = []
    if pronoun_note:
        parts.append(f"Bird pronouns (use these): {pronoun_note}")
    parts.append(f"Question: {question.strip()}")
    if facts:
        parts.append(f"Counts (exact tallies — trust over prose, but never quote):\n{facts}")
    header = f"Memory notes from {window_phrase}" if window_phrase else "Memory notes"
    parts.append(f"{header}:\n{body}")
    messages = [
        {"role": "system", "content": _ACTIVITY_QA_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    # llm_model is an instruct model (gemma3): answer directly, no thinking.
    # With a reasoning model think=True burned the whole 768-token budget on
    # deliberation and returned EMPTY content; num_predict now caps the answer.
    reply = _chat_maybe_streaming(
        client, model, messages, num_predict=400,
        timeout_seconds=timeout_seconds, on_partial=on_partial, cancelled=cancelled,
    )
    return _strip_stray_yesno(clean_reply(reply), question)


def _chat_maybe_streaming(
    client,
    model: str,
    messages: list[dict[str, str]],
    *,
    num_predict: int,
    timeout_seconds: float | None,
    on_partial=None,
    cancelled=None,
) -> str:
    """One chat turn, streamed when the caller wants partials.

    With ``on_partial`` set (and a client that can stream), tokens surface as
    they arrive so the recall answer / report caption grows on screen instead
    of appearing all at once; otherwise this is the plain blocking call.
    """
    if on_partial is not None and hasattr(client, "chat_stream"):
        stream = client.chat_stream(
            model, messages, think=False,
            num_predict=num_predict, timeout_seconds=timeout_seconds,
        )
        raw, _ = collect_stream(stream, on_partial=on_partial, cancelled=cancelled)
        return raw
    return client.chat(
        model, messages, think=False,
        num_predict=num_predict, timeout_seconds=timeout_seconds,
    )


def summarise_activity(
    client,
    model: str,
    notes: list[str],
    subject: str = "",
    pronoun_note: str = "",
    *,
    timeout_seconds: float | None = None,
    on_partial=None,
    cancelled=None,
) -> str:
    """Fold journal memory notes into a terse bulleted activity report.

    ``pronoun_note`` states each bird's sex (the notes often don't carry it, so
    the model would otherwise default everyone to "he").
    """
    if not notes:
        return ""
    body = "\n".join(f"- {line}" for line in notes)
    ask = f"Summarise {subject}'s activity from these notes:\n{body}" if subject else f"Notes:\n{body}"
    if pronoun_note:
        ask = f"Bird pronouns (use these exactly): {pronoun_note}\n\n{ask}"
    # Instruct model (gemma3): think=False means a direct answer, not leaked
    # reasoning; 5 short bullets fit comfortably under this cap.
    reply = _chat_maybe_streaming(
        client,
        model,
        [
            {"role": "system", "content": _ACTIVITY_SUMMARY_PROMPT},
            {"role": "user", "content": ask},
        ],
        num_predict=350,
        timeout_seconds=timeout_seconds,
        on_partial=on_partial,
        cancelled=cancelled,
    )
    return clean_reply(reply)
