"""Stats over the day-memory store, for the ``/memory`` command.

Answers, in one screen: how much history there is (days, entries, per-bird
records), what still needs work (observations awaiting VLM backfill, pre-v3
entries that need ``memory-migrate``, photos that went missing, queued outage
messages), and what it all costs on disk (image count/bytes, memories total,
the whole ``data/server`` tree).

The scan reads every day JSONL and stats every referenced photo — a second or
two on a long history — so the Telegram handler runs it off the poll loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from lib.backfill import MEMORY_BACKFILL_DAYS
from lib.clock import now_ph
from lib.journal import load_day_records


@dataclass
class MemoryStatsReport:
    days: int = 0
    first_day: str = ""
    last_day: str = ""
    entries: int = 0
    observations: int = 0
    bird_records: int = 0  # per-bird MemoryDetection records (the unit of recall)
    activity_tagged: int = 0  # bird records that carry a VLM activity
    # v3 obs with identity+photo but no VLM decoration, split by whether the
    # AUTO backfill worker will pick them up (it only scans the last
    # MEMORY_BACKFILL_DAYS) or they need a manual memory-migrate run.
    undecorated: int = 0
    undecorated_stale: int = 0
    pre_v3_entries: int = 0  # old-format entries; memory-migrate's job
    photos_referenced: int = 0
    photos_missing: int = 0
    image_count: int = 0
    image_bytes: int = 0
    memories_bytes: int = 0
    data_bytes: int = 0
    pending_messages: int = 0


def _tree_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def gather_memory_stats(
    memories_dir: Path,
    *,
    pending_count: int = 0,
    data_dir: Path | None = None,
    now=now_ph,
) -> MemoryStatsReport:
    memories_dir = Path(memories_dir)
    report = MemoryStatsReport(pending_messages=pending_count)
    # The auto backfill worker only scans this far back — an undecorated
    # observation OLDER than that is memory-migrate's job, and calling it
    # "(auto)" in the report would overstate forever.
    backfill_floor = now().date().toordinal() - (MEMORY_BACKFILL_DAYS - 1)

    day_files = sorted(f for f in memories_dir.glob("*.jsonl"))
    days: list[str] = []
    for day_file in day_files:
        try:
            day = date.fromisoformat(day_file.stem)
        except ValueError:
            continue  # not a day file (e.g. stray tmp)
        days.append(day_file.stem)
        for record in load_day_records(memories_dir, day):
            report.entries += 1
            if record.get("version") != 3:
                report.pre_v3_entries += 1
            for obs in record.get("observations") or []:
                if not isinstance(obs, dict):
                    continue
                report.observations += 1
                detections = obs.get("detections") or []
                report.bird_records += len(detections)
                report.activity_tagged += sum(
                    1 for d in detections if isinstance(d, dict) and d.get("activity")
                )
                photo = str(obs.get("photo", ""))
                if photo:
                    report.photos_referenced += 1
                    if not Path(photo).exists():
                        report.photos_missing += 1
                if (
                    record.get("version") == 3
                    and not obs.get("vlm_model")
                    and detections
                    and photo
                ):
                    if day.toordinal() >= backfill_floor:
                        report.undecorated += 1
                    else:
                        report.undecorated_stale += 1
    report.days = len(days)
    report.first_day = days[0] if days else ""
    report.last_day = days[-1] if days else ""

    images_dir = memories_dir / "images"
    if images_dir.exists():
        for path in images_dir.iterdir():
            try:
                if path.is_file():
                    report.image_count += 1
                    report.image_bytes += path.stat().st_size
            except OSError:
                continue
    report.memories_bytes = _tree_size(memories_dir)
    if data_dir is not None:
        report.data_bytes = _tree_size(Path(data_dir))
    return report


def _fmt_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def format_memory_stats(report: MemoryStatsReport) -> str:
    span = ""
    if report.first_day:
        span = (
            f" ({report.first_day} → {report.last_day})"
            if report.first_day != report.last_day
            else f" ({report.first_day})"
        )
    tagged_pct = (
        f" · {report.activity_tagged / report.bird_records:.0%} activity-tagged"
        if report.bird_records
        else ""
    )
    lines = [
        "🧠 Memory",
        f"Days: {report.days}{span}",
        f"Entries: {report.entries:,} · observations {report.observations:,} · "
        f"bird records {report.bird_records:,}{tagged_pct}",
    ]

    todo: list[str] = []
    if report.undecorated:
        todo.append(f"{report.undecorated:,} awaiting VLM backfill (auto)")
    if report.undecorated_stale:
        todo.append(f"{report.undecorated_stale:,} older undecorated (run memory-migrate)")
    if report.pre_v3_entries:
        todo.append(f"{report.pre_v3_entries:,} pre-v3 (run memory-migrate)")
    if report.pending_messages:
        todo.append(f"{report.pending_messages:,} queued message(s)")
    lines.append("To do: " + (" · ".join(todo) if todo else "all caught up ✅"))

    lines.append(
        f"Photos: {report.image_count:,} images · {_fmt_bytes(report.image_bytes)}"
    )
    if report.photos_missing:
        lines.append(
            f"⚠️ {report.photos_missing:,} of {report.photos_referenced:,} referenced "
            "photo file(s) missing"
        )
    storage = f"Storage: memories {_fmt_bytes(report.memories_bytes)}"
    if report.data_bytes:
        storage += f" · data/server total {_fmt_bytes(report.data_bytes)}"
    lines.append(storage)
    return "\n".join(lines)
