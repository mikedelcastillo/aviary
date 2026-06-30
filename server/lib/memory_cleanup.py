"""Migrate legacy Markdown memories into structured memory records.

The current memory writer stores two representations:

* ``YYYY-MM-DD.md`` remains human-readable.
* ``YYYY-MM-DD.jsonl`` stores per-camera observations for precise Q&A.

This module backfills that shape for older Markdown-only days. It is deliberately
conservative: when a legacy caption says no bird was visible, the observation is
kept as uncertainty but is not counted as bird-specific evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from lib.journal import MemoryEntry, MemoryObservation, load_entries, memory_jsonl_path, memory_path
from lib.labels import pretty


_OBS_PREFIX_RE = re.compile(
    r"^(?P<birds>[A-Z][A-Za-z0-9_ ]*(?:, [A-Z][A-Za-z0-9_ ]*)*) on (?P<camera>[^:]{1,80}): (?P<note>.*)$"
)
_OBS_SPLIT_RE = re.compile(r";\s+(?=[A-Z][A-Za-z0-9_ ]*(?:, [A-Z][A-Za-z0-9_ ]*)* on [^:]{1,80}:)")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\u2022)\s+(.*?)(?:\s{2,})?$")
_PRONOUN_MARK_RE = re.compile(r"\s*\((?:he|she|they|him|her|them)\)\s*", re.IGNORECASE)
_UNCERTAIN_RE = re.compile(
    r"\b(?:no (?:clear )?birds?|not (?:clearly )?visible|not visible|empty|detector found no birds?|"
    r"no clear view|no birds visible|no birds detected|not present|no presence)\b",
    re.IGNORECASE,
)
_KNOWN_LABELS = {
    "bambi",
    "budgie",
    "cockatiel",
    "draft",
    "jynx",
    "lovebird",
    "matcha",
    "percy",
    "pizza",
    "unknown_bird",
}
_ALIASES = {
    "unknown bird": "unknown_bird",
}


@dataclass
class CleanupStats:
    files: int = 0
    entries: int = 0
    observations: int = 0
    uncertain_observations: int = 0
    bird_labels_removed: int = 0
    jsonl_written: int = 0


def _label_from_text(text: str) -> str | None:
    cleaned = _PRONOUN_MARK_RE.sub(" ", text).strip().lower().replace("_", " ")
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    label = cleaned.replace(" ", "_")
    if label in _KNOWN_LABELS:
        return label
    return None


def _labels_from_prefix(text: str) -> list[str]:
    labels: list[str] = []
    for raw in text.split(","):
        label = _label_from_text(raw)
        if label is not None and label not in labels:
            labels.append(label)
    return labels


def _labels_mentioned(text: str, candidates: list[str]) -> list[str]:
    lowered = text.lower()
    labels: list[str] = []
    for label in candidates:
        token = re.escape(label.replace("_", " "))
        pattern = rf"(?<![a-z]){token}(?!['\u2019]s)(?![a-z])"
        underscored = re.escape(label)
        alt_pattern = rf"(?<![a-z]){underscored}(?!['\u2019]s)(?![a-z])"
        if re.search(pattern, lowered) or re.search(alt_pattern, lowered):
            labels.append(label)
    return labels


def _is_uncertain(text: str) -> bool:
    return bool(_UNCERTAIN_RE.search(text))


def _clean_note(note: str) -> str:
    note = note.strip()
    note = re.sub(r"\s+", " ", note)
    return note.strip(" -")


def _split_legacy_segments(note: str) -> list[str]:
    stripped = note.strip()
    if not stripped:
        return []
    bullets = [_clean_note(match.group(1)) for match in _BULLET_RE.finditer(stripped)]
    if bullets:
        return [b for b in bullets if b]
    return [_clean_note(part) for part in _OBS_SPLIT_RE.split(stripped) if _clean_note(part)]


def _observation_text(observation: MemoryObservation) -> str:
    note = _clean_note(observation.note)
    if observation.birds:
        who = ", ".join(pretty(b) for b in observation.birds)
        if observation.camera:
            return f"{who} on {observation.camera}: {note}"
        return f"{who}: {note}"
    if observation.camera:
        return f"Unconfirmed on {observation.camera}: {note}"
    return f"Unconfirmed: {note}"


def _entry_record(entry: MemoryEntry) -> dict:
    return {
        "version": 1,
        "time": entry.time.isoformat(),
        "birds": entry.birds,
        "note": entry.note,
        "photos": entry.photos,
        "observations": [
            {
                "camera": observation.camera,
                "birds": observation.birds,
                "note": observation.note,
                "photo": observation.photo,
            }
            for observation in entry.observations
        ],
    }


def _migrate_entry(entry: MemoryEntry) -> tuple[MemoryEntry, int, int]:
    if entry.observations:
        observations = [
            MemoryObservation(
                camera=observation.camera,
                birds=[] if _is_uncertain(observation.note) else observation.birds,
                note=_clean_note(observation.note),
                photo=observation.photo,
            )
            for observation in entry.observations
        ]
        birds = sorted({bird for observation in observations for bird in observation.birds})
        note_lines = [_observation_text(observation) for observation in observations]
        note = note_lines[0] if len(note_lines) == 1 else "\n".join(f"- {line}" for line in note_lines)
        removed = sum(
            len(observation.birds)
            for observation in entry.observations
            if _is_uncertain(observation.note)
        )
        uncertain = sum(1 for observation in entry.observations if _is_uncertain(observation.note))
        return MemoryEntry(entry.time, birds, note, entry.photos, observations), uncertain, removed

    segments = _split_legacy_segments(entry.note)
    if not segments:
        segments = [entry.note.strip() or "(activity)"]
    observations: list[MemoryObservation] = []
    uncertain_count = 0
    removed_count = 0
    for index, segment in enumerate(segments):
        match = _OBS_PREFIX_RE.match(segment)
        camera = ""
        birds: list[str]
        note = segment
        if match:
            birds = _labels_from_prefix(match.group("birds"))
            camera = match.group("camera").strip()
            note = match.group("note").strip()
        else:
            candidates = sorted(set(entry.birds) | _KNOWN_LABELS)
            birds = _labels_mentioned(segment, candidates)
            if not birds and len(segments) == 1:
                birds = list(entry.birds)
            if not birds and len(entry.birds) == 1:
                birds = list(entry.birds)
        if _is_uncertain(note):
            removed_count += len(birds)
            birds = []
            uncertain_count += 1
        photo = entry.photos[index] if index < len(entry.photos) else ""
        observations.append(
            MemoryObservation(
                camera=camera,
                birds=birds,
                note=_clean_note(note),
                photo=photo,
            )
        )
    birds = sorted({bird for observation in observations for bird in observation.birds})
    note_lines = [_observation_text(observation) for observation in observations]
    note = note_lines[0] if len(note_lines) == 1 else "\n".join(f"- {line}" for line in note_lines)
    return (
        MemoryEntry(entry.time, birds, note, entry.photos, observations),
        uncertain_count,
        removed_count,
    )


def _render_markdown(day: date, entries: list[MemoryEntry]) -> str:
    chunks = [f"# Aviary memories — {day.isoformat()}", ""]
    for entry in entries:
        birds = ", ".join(entry.birds) if entry.birds else "quiet"
        chunks.append(f"## {entry.time.strftime('%H:%M')} | {birds}")
        chunks.append(entry.note.strip() or "(activity)")
        for photo in entry.photos:
            chunks.append(f"> photo: {photo}")
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def _backup_file(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)


def migrate_memories(memories_dir: Path, *, dry_run: bool = False, backup: bool = True) -> CleanupStats:
    memories_dir = Path(memories_dir)
    stats = CleanupStats()
    backup_dir = memories_dir / f".cleanup-backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for md_path in sorted(memories_dir.glob("????-??-??.md")):
        try:
            day = date.fromisoformat(md_path.stem)
        except ValueError:
            continue
        entries = load_entries(memories_dir, day)
        migrated: list[MemoryEntry] = []
        for entry in entries:
            new_entry, uncertain, removed = _migrate_entry(entry)
            migrated.append(new_entry)
            stats.entries += 1
            stats.observations += len(new_entry.observations)
            stats.uncertain_observations += uncertain
            stats.bird_labels_removed += removed
        stats.files += 1
        if dry_run:
            continue
        jsonl_path = memory_jsonl_path(memories_dir, day)
        if backup:
            _backup_file(md_path, backup_dir)
            _backup_file(jsonl_path, backup_dir)
        memory_path(memories_dir, day).write_text(_render_markdown(day, migrated), encoding="utf-8")
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for entry in migrated:
                handle.write(json.dumps(_entry_record(entry), ensure_ascii=False) + "\n")
                stats.jsonl_written += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean legacy aviary memory Markdown and backfill JSONL")
    parser.add_argument("memories_dir", nargs="?", default="data/server/memories")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    parser.add_argument("--no-backup", action="store_true", help="rewrite files without creating backup copies")
    args = parser.parse_args(argv)
    stats = migrate_memories(Path(args.memories_dir), dry_run=args.dry_run, backup=not args.no_backup)
    mode = "would clean" if args.dry_run else "cleaned"
    print(
        f"{mode} {stats.files} file(s), {stats.entries} entries, {stats.observations} observations; "
        f"{stats.uncertain_observations} uncertain observations, "
        f"{stats.bird_labels_removed} bird labels removed from uncertain observations, "
        f"{stats.jsonl_written} JSONL records written."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
