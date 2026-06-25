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
from lib.ai.vlm import build_detection_context, describe_image
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
    """Minimal duck-typed detection for build_detection_context."""

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


def caption_sighting(
    client,
    vlm_model: str,
    sighting: Sighting,
    member_species: dict[str, str] | None = None,
    pronouns: dict[str, str] | None = None,
    *,
    timeout_seconds: float | None = None,
) -> str:
    """A short VLM caption for one sighting, grounded by its detection box."""
    context = build_detection_context(
        [_BoxDetection(sighting.label, sighting.bbox)],
        sighting.width,
        sighting.height,
        member_species,
        pronouns,
    )
    return describe_image(
        client,
        vlm_model,
        sighting.path.read_bytes(),
        "In one short sentence, say what this bird is doing.",
        context=context,
        timeout_seconds=timeout_seconds,
    )


DIGEST_SYSTEM_PROMPT = (
    "You are the warm, upbeat caretaker of a home aviary sending a short 'daycare "
    "update' text about the pet birds. Given timestamped observations, write 2-4 "
    "friendly sentences: who was active, who was together, what they were up to "
    "(eating, playing, preening, resting). Use each bird's pronoun exactly as it "
    "appears in the notes (he/she). Don't invent things not in the notes. No "
    "lists, no preamble — just the update."
)


def summarise_day(
    client, llm_model: str, observations: list[str], *, header: str = "", timeout_seconds: float | None = None
) -> str:
    """Fold per-photo captions into one warm caretaker summary via the LLM."""
    if not observations:
        return ""
    notes = "\n".join(f"- {line}" for line in observations)
    user = (f"{header}\n" if header else "") + f"Observations:\n{notes}"
    reply = client.chat(
        llm_model,
        [
            {"role": "system", "content": DIGEST_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        think=True,
        timeout_seconds=timeout_seconds,
    )
    return strip_thinking(reply)
