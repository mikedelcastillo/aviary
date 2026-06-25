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

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


LOGGER = logging.getLogger("lib.journal")

_ENTRY_RE = re.compile(
    r"## (\d{2}:\d{2}) \| ([^\n]*)\n(.*?)(?=\n## |\Z)",
    re.DOTALL,
)
_PHOTO_RE = re.compile(r"^> photo: (.+)$", re.MULTILINE)


@dataclass
class MemoryEntry:
    time: datetime
    birds: list[str]
    note: str
    photos: list[str] = field(default_factory=list)


def memory_path(memories_dir: Path, day: date) -> Path:
    return memories_dir / f"{day.isoformat()}.md"


def append_entry(memories_dir: Path, entry: MemoryEntry) -> Path:
    """Append one memory entry to its day's file (creating it with a header)."""
    memories_dir.mkdir(parents=True, exist_ok=True)
    path = memory_path(memories_dir, entry.time.date())
    new_file = not path.exists()
    birds = ", ".join(entry.birds) if entry.birds else "quiet"
    block = f"## {entry.time.strftime('%H:%M')} | {birds}\n{entry.note.strip()}\n"
    for photo in entry.photos:
        block += f"> photo: {photo}\n"
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(f"# Aviary memories — {entry.time.date().isoformat()}\n\n")
        handle.write(block + "\n")
    return path


def load_entries(memories_dir: Path, day: date) -> list[MemoryEntry]:
    """Parse a day's memory file into entries (empty if the file is absent)."""
    path = memory_path(memories_dir, day)
    if not path.exists():
        return []
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
    return entries


def load_recent(
    memories_dir: Path,
    since: datetime,
    until: datetime,
    birds: set[str] | None = None,
) -> list[MemoryEntry]:
    """Entries in ``[since, until]`` across the relevant day file(s), newest last.

    ``birds`` (lowercased) filters to entries mentioning at least one of them.
    """
    entries: list[MemoryEntry] = []
    # A range almost always spans one or two day files.
    for day in {since.date(), until.date()}:
        for entry in load_entries(memories_dir, day):
            if since <= entry.time <= until:
                entries.append(entry)
    if birds:
        entries = [e for e in entries if any(b in birds for b in e.birds)]
    entries.sort(key=lambda e: e.time)
    return entries
