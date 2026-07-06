"""Persistent per-day memory of bird activity, as appendable Markdown.

The caretaker keeps a running journal at ``data/server/memories/YYYY-MM-DD.md``.
Every time it analyses what the birds are up to (on a schedule, or when something
notable happens) it appends a timestamped entry — which birds, a short VLM-written
note, and the photo it looked at. That file is the day's durable memory: the
``/activity`` command and natural questions ("what did percy do today?") read it
back, and it survives restarts.

The format stays human-readable Markdown but is parseable: one ``## HH:MM | birds``
section per entry, the note beneath, and a ``> photo: <path>`` line.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


LOGGER = logging.getLogger("lib.journal")

# Serializes every write to the day files within this process: the live
# MemoryMaker APPENDS while the backfill worker REWRITES (read-modify-write) the
# same file — unserialized, an append landing mid-rewrite would be lost.
_DAY_FILE_LOCK = threading.Lock()

_ENTRY_RE = re.compile(
    r"## (\d{2}:\d{2}) \| ([^\n]*)\n(.*?)(?=\n## |\Z)",
    re.DOTALL,
)
_PHOTO_RE = re.compile(r"^> photo: (.+)$", re.MULTILINE)
# Leading markdown header hashes on a note line ("## ", "# ") — stripped on write
# so a note can't forge an entry header (see append_entry).
_HEADER_HASHES = re.compile(r"(?m)^#+[ \t]+")


@dataclass
class MemoryDetection:
    """One bird's record within an observation — the v3 structured unit of recall.

    Identity (``label``/``confidence``/``bbox``) comes from YOLO; the behaviour
    fields (``activity``/``tags``/``posture``/``health``) come from the VLM keyed to
    that bird's labeled box, so activity is attributed to the RIGHT bird. ``bbox`` is
    xyxy in the coordinates of the saved (annotated) photo. Storing all of this makes
    the memory queryable — filter by confidence, count activities, find co-occurrence
    — without re-parsing prose or re-running the models. Behaviour fields are empty on
    pre-v3 records (back-compat).
    """

    label: str = ""
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    activity: str = ""  # primary activity, e.g. "preening" (from the VLM)
    tags: list[str] = field(default_factory=list)  # all activity tags seen
    posture: str = ""  # e.g. "perched", "on the cage floor"
    health: str = ""  # a visible concern ("fluffed", "lethargic"), else ""


@dataclass
class MemoryObservation:
    camera: str = ""
    birds: list[str] = field(default_factory=list)
    note: str = ""
    photo: str = ""
    detections: list[MemoryDetection] = field(default_factory=list)
    # Provenance: which models produced this observation, so we know what to trust
    # and which records to re-process when a model changes. Per-bird detector
    # confidence lives on each MemoryDetection.
    detector_model: str = ""  # e.g. "live-019.pt"
    vlm_model: str = ""  # e.g. "qwen2.5vl:3b"


@dataclass
class MemoryEntry:
    time: datetime
    birds: list[str]
    note: str
    photos: list[str] = field(default_factory=list)
    observations: list[MemoryObservation] = field(default_factory=list)


def memory_path(memories_dir: Path, day: date) -> Path:
    return memories_dir / f"{day.isoformat()}.md"


def memory_jsonl_path(memories_dir: Path, day: date) -> Path:
    return memories_dir / f"{day.isoformat()}.jsonl"


def _clean_label(label: str) -> str:
    return str(label).strip().lower()


def _clean_bbox(raw) -> tuple[int, int, int, int]:
    try:
        values = [int(round(float(v))) for v in raw][:4]
    except (TypeError, ValueError):
        return (0, 0, 0, 0)
    if len(values) != 4:
        return (0, 0, 0, 0)
    return (values[0], values[1], values[2], values[3])


def _clean_detection(detection: MemoryDetection) -> MemoryDetection:
    return MemoryDetection(
        label=_clean_label(detection.label),
        confidence=round(float(detection.confidence or 0.0), 3),
        bbox=_clean_bbox(detection.bbox),
        activity=str(detection.activity or "").strip().lower(),
        tags=sorted({str(t).strip().lower() for t in detection.tags if str(t).strip()}),
        posture=str(detection.posture or "").strip().lower(),
        health=str(detection.health or "").strip().lower(),
    )


def _clean_observation(observation: MemoryObservation) -> MemoryObservation:
    return MemoryObservation(
        camera=str(observation.camera).strip(),
        birds=sorted({_clean_label(b) for b in observation.birds if str(b).strip()}),
        note=str(observation.note).strip(),
        photo=str(observation.photo).strip(),
        detections=[_clean_detection(d) for d in observation.detections if str(d.label).strip()],
        detector_model=str(observation.detector_model).strip(),
        vlm_model=str(observation.vlm_model).strip(),
    )


def _detection_record(d: MemoryDetection) -> dict:
    # Behaviour fields are omitted when empty so pre-v3 (identity-only) detections
    # stay compact on disk.
    record: dict = {"label": d.label, "confidence": d.confidence, "bbox": list(d.bbox)}
    if d.activity:
        record["activity"] = d.activity
    if d.tags:
        record["tags"] = d.tags
    if d.posture:
        record["posture"] = d.posture
    if d.health:
        record["health"] = d.health
    return record


def observation_record(observation: MemoryObservation) -> dict:
    """The on-disk JSONL dict for one (cleaned) observation."""
    obs = _clean_observation(observation)
    return {
        "camera": obs.camera,
        "birds": obs.birds,
        "note": obs.note,
        "photo": obs.photo,
        "detections": [_detection_record(d) for d in obs.detections],
        **({"detector_model": obs.detector_model} if obs.detector_model else {}),
        **({"vlm_model": obs.vlm_model} if obs.vlm_model else {}),
    }


def _entry_record(entry: MemoryEntry) -> dict:
    return {
        "version": 3,
        "time": entry.time.isoformat(),
        "birds": sorted({_clean_label(b) for b in entry.birds if str(b).strip()}),
        "note": entry.note.strip(),
        "photos": [str(photo) for photo in entry.photos],
        "observations": [
            record
            for record in (observation_record(o) for o in entry.observations)
            if record["note"] or record["birds"] or record["photo"] or record["camera"]
        ],
    }


def _entry_from_record(data: dict) -> MemoryEntry | None:
    try:
        when = datetime.fromisoformat(str(data["time"]))
    except Exception:
        return None
    observations: list[MemoryObservation] = []
    for raw in data.get("observations") or []:
        if not isinstance(raw, dict):
            continue
        observations.append(
            _clean_observation(
                MemoryObservation(
                    camera=str(raw.get("camera", "")),
                    birds=[_clean_label(b) for b in raw.get("birds", []) if str(b).strip()],
                    note=str(raw.get("note", "")),
                    photo=str(raw.get("photo", "")),
                    detector_model=str(raw.get("detector_model", "")),
                    vlm_model=str(raw.get("vlm_model", "")),
                    # Absent on v1 records (pre-enrichment) — back-compat via default [].
                    detections=[
                        MemoryDetection(
                            label=str(d.get("label", "")),
                            confidence=float(d.get("confidence", 0.0) or 0.0),
                            bbox=_clean_bbox(d.get("bbox")),
                            activity=str(d.get("activity", "")),
                            tags=[str(t) for t in (d.get("tags") or []) if str(t).strip()],
                            posture=str(d.get("posture", "")),
                            health=str(d.get("health", "")),
                        )
                        for d in (raw.get("detections") or [])
                        if isinstance(d, dict)
                    ],
                )
            )
        )
    return MemoryEntry(
        time=when,
        birds=sorted({_clean_label(b) for b in data.get("birds", []) if str(b).strip()}),
        note=str(data.get("note", "")).strip(),
        photos=[str(photo) for photo in data.get("photos", []) if str(photo).strip()],
        observations=observations,
    )


def _load_jsonl_entries(memories_dir: Path, day: date) -> list[MemoryEntry]:
    path = memory_jsonl_path(memories_dir, day)
    if not path.exists():
        return []
    entries: list[MemoryEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.debug("Skipping malformed memory JSONL record in %s", path)
            continue
        if not isinstance(data, dict):
            continue
        entry = _entry_from_record(data)
        if entry is not None and entry.time.date() == day:
            entries.append(entry)
    return entries


def _entry_key(entry: MemoryEntry) -> tuple:
    minute = entry.time.replace(second=0, microsecond=0)
    return (
        minute,
        tuple(sorted(entry.birds)),
        entry.note.strip(),
        tuple(entry.photos),
    )


def append_entry(memories_dir: Path, entry: MemoryEntry) -> Path:
    """Append one memory entry to its day's file (creating it with a header)."""
    memories_dir.mkdir(parents=True, exist_ok=True)
    path = memory_path(memories_dir, entry.time.date())
    birds = ", ".join(entry.birds) if entry.birds else "quiet"
    # A VLM note whose line starts with "## " would imitate an entry header and
    # corrupt parsing on read-back (it could be misread as a new entry). Strip
    # leading markdown header hashes per line so a note can never forge a header.
    note = _HEADER_HASHES.sub("", entry.note.strip())
    block = f"## {entry.time.strftime('%H:%M')} | {birds}\n{note}\n"
    for photo in entry.photos:
        block += f"> photo: {photo}\n"
    record_entry = MemoryEntry(entry.time, entry.birds, note, entry.photos, entry.observations)
    with _DAY_FILE_LOCK:
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(f"# Aviary memories — {entry.time.date().isoformat()}\n\n")
            handle.write(block + "\n")
        with memory_jsonl_path(memories_dir, entry.time.date()).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_entry_record(record_entry), ensure_ascii=False) + "\n")
    return path


def load_day_records(memories_dir: Path, day: date) -> list[dict]:
    """A day's raw JSONL records (dicts), skipping malformed lines.

    The raw-dict view (rather than parsed :class:`MemoryEntry`) is what the
    backfill scanner needs: it must see which observations LACK a ``vlm_model``
    key, and must round-trip unknown fields untouched.
    """
    path = memory_jsonl_path(memories_dir, day)
    if not path.exists():
        return []
    # Read under the day-file lock: an in-flight append's buffered flush can
    # land mid-line, and an unlocked reader would see (and drop) the torn tail.
    with _DAY_FILE_LOCK:
        text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def update_day_observations(
    memories_dir: Path, day: date, updates: dict[str, dict[str, dict]]
) -> int:
    """Replace specific observation records in a day's JSONL, atomically.

    ``updates`` maps an entry's exact ``time`` ISO string to ``{photo_path:
    new_observation_record}``. ONLY the matched observations are swapped; the
    entry-level ``time``/``birds``/``note``/``photos`` stay byte-identical —
    they form the key ``load_entries`` uses to merge the JSONL with the
    Markdown day file, so changing them would duplicate the entry on read-back.

    The file is re-read and rewritten under the day-file lock (tmp + atomic
    replace), so a live append landing around the rewrite is never lost.
    Returns the number of observations replaced.
    """
    path = memory_jsonl_path(memories_dir, day)
    if not updates or not path.exists():
        return 0
    replaced = 0
    with _DAY_FILE_LOCK:
        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                lines.append(line)  # never drop a line we can't parse
                continue
            if not isinstance(record, dict):
                lines.append(line)
                continue
            by_photo = updates.get(str(record.get("time", "")))
            if by_photo:
                observations = record.get("observations") or []
                for i, obs in enumerate(observations):
                    if not isinstance(obs, dict) or obs.get("vlm_model"):
                        continue
                    new_obs = by_photo.get(str(obs.get("photo", "")))
                    if new_obs is not None:
                        observations[i] = new_obs
                        replaced += 1
            lines.append(json.dumps(record, ensure_ascii=False))
        if replaced:
            # PID-unique tmp name so a concurrently-running memory-migrate (a
            # separate process) can never promote or collide with our tmp file.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.replace(path)
    return replaced


def load_entries(memories_dir: Path, day: date) -> list[MemoryEntry]:
    """Parse a day's memory file into entries (empty if the file is absent)."""
    path = memory_path(memories_dir, day)
    json_entries = _load_jsonl_entries(memories_dir, day)
    if not path.exists():
        return sorted(json_entries, key=lambda e: e.time)
    text = path.read_text(encoding="utf-8")
    entries: list[MemoryEntry] = []
    for match in _ENTRY_RE.finditer(text):
        hhmm, birds_raw, body = match.group(1), match.group(2), match.group(3)
        photos = [p.strip() for p in _PHOTO_RE.findall(body)]
        note = _PHOTO_RE.sub("", body).strip()
        try:
            when = datetime.combine(
                day, datetime.strptime(hhmm, "%H:%M").time()
            )
        except ValueError:
            continue
        birds = [b.strip().lower() for b in birds_raw.split(",") if b.strip() and b.strip() != "quiet"]
        entries.append(MemoryEntry(time=when, birds=birds, note=note, photos=photos))
    if json_entries:
        by_key: dict[tuple, list[MemoryEntry]] = {}
        for entry in json_entries:
            by_key.setdefault(_entry_key(entry), []).append(entry)
        merged: list[MemoryEntry] = []
        for entry in entries:
            matches = by_key.get(_entry_key(entry))
            merged.append(matches.pop(0) if matches else entry)
        extras = [entry for matches in by_key.values() for entry in matches]
        entries = merged + extras
        entries.sort(key=lambda e: e.time)
    return entries


def load_recent(
    memories_dir: Path,
    since: datetime,
    until: datetime,
    birds: set[str] | None = None,
) -> list[MemoryEntry]:
    """Entries in ``[since, until]`` across the relevant day file(s), newest last.

    ``birds`` (lowercased) filters to entries mentioning at least one of them.
    Spans every day in the range, so a week-long window reads all seven files.
    """
    entries: list[MemoryEntry] = []
    day = since.date()
    while day <= until.date():
        for entry in load_entries(memories_dir, day):
            if since <= entry.time <= until:
                entries.append(entry)
        day += timedelta(days=1)
    if birds:
        entries = [e for e in entries if any(b in birds for b in e.birds)]
    entries.sort(key=lambda e: e.time)
    return entries


def humanize_ago(then: datetime, now: datetime) -> str:
    """A relative phrase for how long ago ``then`` was: "12 minutes ago", "2 hours ago"."""
    seconds = (now - then).total_seconds()
    if seconds < 60:
        return "just now"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(round(minutes / 60))
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(round(hours / 24))
    return f"{days} day{'s' if days != 1 else ''} ago"
