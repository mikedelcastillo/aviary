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
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


LOGGER = logging.getLogger("lib.journal")

_ENTRY_RE = re.compile(
    r"## (\d{2}:\d{2}) \| ([^\n]*)\n(.*?)(?=\n## |\Z)",
    re.DOTALL,
)
_PHOTO_RE = re.compile(r"^> photo: (.+)$", re.MULTILINE)
# Leading markdown header hashes on a note line ("## ", "# ") — stripped on write
# so a note can't forge an entry header (see append_entry).
_HEADER_HASHES = re.compile(r"(?m)^#+[ \t]+")


@dataclass
class MemoryObservation:
    camera: str = ""
    birds: list[str] = field(default_factory=list)
    note: str = ""
    photo: str = ""


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


def _clean_observation(observation: MemoryObservation) -> MemoryObservation:
    return MemoryObservation(
        camera=str(observation.camera).strip(),
        birds=sorted({_clean_label(b) for b in observation.birds if str(b).strip()}),
        note=str(observation.note).strip(),
        photo=str(observation.photo).strip(),
    )


def _entry_record(entry: MemoryEntry) -> dict:
    return {
        "version": 1,
        "time": entry.time.isoformat(),
        "birds": sorted({_clean_label(b) for b in entry.birds if str(b).strip()}),
        "note": entry.note.strip(),
        "photos": [str(photo) for photo in entry.photos],
        "observations": [
            {
                "camera": obs.camera,
                "birds": obs.birds,
                "note": obs.note,
                "photo": obs.photo,
            }
            for obs in (_clean_observation(o) for o in entry.observations)
            if obs.note or obs.birds or obs.photo or obs.camera
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
    new_file = not path.exists()
    birds = ", ".join(entry.birds) if entry.birds else "quiet"
    # A VLM note whose line starts with "## " would imitate an entry header and
    # corrupt parsing on read-back (it could be misread as a new entry). Strip
    # leading markdown header hashes per line so a note can never forge a header.
    note = _HEADER_HASHES.sub("", entry.note.strip())
    block = f"## {entry.time.strftime('%H:%M')} | {birds}\n{note}\n"
    for photo in entry.photos:
        block += f"> photo: {photo}\n"
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(f"# Aviary memories — {entry.time.date().isoformat()}\n\n")
        handle.write(block + "\n")
    record_entry = MemoryEntry(entry.time, entry.birds, note, entry.photos, entry.observations)
    with memory_jsonl_path(memories_dir, entry.time.date()).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_entry_record(record_entry), ensure_ascii=False) + "\n")
    return path


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
