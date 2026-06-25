"""Sleep records + persistence for the bird sleep tracker (pure, no threads).

One :class:`SleepNight` is recorded per night — the flock's shared dark window
(the room darkens and lightens together; IR feeds can't resolve individual birds
in the dark, so this is honestly a ROOM/FLOCK metric). Finalized nights are
appended as JSON lines to monthly files under ``data/server/sleep/YYYY-MM.jsonl``
(structured numeric data the score/report consume, mirroring how
:mod:`lib.journal` persists day memory). The single in-progress night is mirrored
to ``_current.json`` so a mid-night restart resumes it.

Everything here is pure given a directory: no clock, no threads, no I/O beyond
the path handed in — so it unit-tests directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

LOGGER = logging.getLogger("lib.sleep.model")

CURRENT_FILENAME = "_current.json"

# Disturbance kinds.
LIGHT = "light"
MOTION = "motion"
NIGHT_FRIGHT = "night_fright"


@dataclass(frozen=True)
class Disturbance:
    """One discrete night disturbance on the dark-window timeline.

    ``minutes`` is how long it lasted — for a LIGHT event, how long the room was
    lit (drives the darkness sub-score); 0 for an instantaneous motion/fright.
    """

    time: datetime
    kind: str  # LIGHT | MOTION | NIGHT_FRIGHT
    detail: str = ""
    minutes: float = 0.0


@dataclass
class SleepNight:
    """The flock's sleep for one night, keyed by the date of its evening.

    A night spanning 20:30 Jun-25 -> 07:10 Jun-26 has ``night_of = 2026-06-25``,
    so "last night" is unambiguous and a month-spanning night files to one month.
    """

    night_of: date
    lights_out: datetime | None = None
    first_light: datetime | None = None
    # Finalized dark-window length: (first_light - lights_out) MINUS any mid-night
    # re-lit intervals, so a light left on doesn't inflate the duration.
    dark_minutes: int | None = None
    disturbances: list[Disturbance] = field(default_factory=list)
    max_motion: float = 0.0
    camera_count_at_dark: int = 0
    camera_count_at_wake: int = 0
    # True when a transition fell back to a sun anchor instead of observed light,
    # or camera membership changed between dark and wake (lowers confidence).
    partial_coverage: bool = False
    score: int | None = None
    components: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    # Whether consistency was scored against a real rolling baseline (False on a
    # cold start with too little history) — so the report doesn't claim a bird is
    # "off their usual" when there is no usual yet.
    baselined: bool = True
    notes: list[str] = field(default_factory=list)
    summary: str = ""
    finalized: bool = False

    @property
    def dark_hours(self) -> float:
        return (self.dark_minutes or 0) / 60.0


# -- (de)serialisation -------------------------------------------------------


def to_json(night: SleepNight) -> dict:
    """A JSON-serialisable dict for one night (datetimes as naive ISO strings)."""
    return {
        "night_of": night.night_of.isoformat(),
        "lights_out": night.lights_out.isoformat() if night.lights_out else None,
        "first_light": night.first_light.isoformat() if night.first_light else None,
        "dark_minutes": night.dark_minutes,
        "disturbances": [
            {"time": d.time.isoformat(), "kind": d.kind, "detail": d.detail, "minutes": d.minutes}
            for d in night.disturbances
        ],
        "max_motion": night.max_motion,
        "camera_count_at_dark": night.camera_count_at_dark,
        "camera_count_at_wake": night.camera_count_at_wake,
        "partial_coverage": night.partial_coverage,
        "score": night.score,
        "components": night.components,
        "confidence": night.confidence,
        "baselined": night.baselined,
        "notes": night.notes,
        "summary": night.summary,
        "finalized": night.finalized,
    }


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def from_json(data: dict) -> SleepNight:
    """Inverse of :func:`to_json`."""
    return SleepNight(
        night_of=date.fromisoformat(data["night_of"]),
        lights_out=_dt(data.get("lights_out")),
        first_light=_dt(data.get("first_light")),
        dark_minutes=data.get("dark_minutes"),
        disturbances=[
            Disturbance(
                time=datetime.fromisoformat(d["time"]),
                kind=d["kind"],
                detail=d.get("detail", ""),
                minutes=float(d.get("minutes", 0.0)),
            )
            for d in data.get("disturbances", [])
        ],
        max_motion=float(data.get("max_motion", 0.0)),
        camera_count_at_dark=int(data.get("camera_count_at_dark", 0)),
        camera_count_at_wake=int(data.get("camera_count_at_wake", 0)),
        partial_coverage=bool(data.get("partial_coverage", False)),
        score=data.get("score"),
        components=dict(data.get("components", {})),
        confidence=float(data.get("confidence", 1.0)),
        baselined=bool(data.get("baselined", True)),
        notes=list(data.get("notes", [])),
        summary=str(data.get("summary", "")),
        finalized=bool(data.get("finalized", False)),
    )


# -- finalized-night store (monthly JSONL) -----------------------------------


def _month_file(sleep_dir: Path, day: date) -> Path:
    return sleep_dir / f"{day.year:04d}-{day.month:02d}.jsonl"


def append_night(sleep_dir: Path, night: SleepNight) -> Path:
    """Append one finalized night as a JSON line to its month's file."""
    sleep_dir.mkdir(parents=True, exist_ok=True)
    path = _month_file(sleep_dir, night.night_of)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_json(night), ensure_ascii=False) + "\n")
    return path


def _read_month(path: Path) -> list[SleepNight]:
    nights: list[SleepNight] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                nights.append(from_json(json.loads(line)))
            except Exception:
                LOGGER.debug("Skipping unreadable sleep line in %s", path, exc_info=True)
    except FileNotFoundError:
        return []
    return nights


def _all_nights(sleep_dir: Path) -> list[SleepNight]:
    if not sleep_dir.exists():
        return []
    nights: list[SleepNight] = []
    for path in sleep_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].jsonl"):
        nights.extend(_read_month(path))
    # Newest last by the evening date then the actual lights-out instant.
    nights.sort(key=lambda n: (n.night_of, n.lights_out or datetime.min))
    return nights


def load_recent(sleep_dir: Path, count: int) -> list[SleepNight]:
    """The last ``count`` finalized nights, newest first."""
    nights = [n for n in _all_nights(sleep_dir) if n.finalized]
    return list(reversed(nights))[:count]


def load_nights(sleep_dir: Path, since: date, until: date) -> list[SleepNight]:
    """Finalized nights with ``since <= night_of <= until``, oldest first."""
    return [n for n in _all_nights(sleep_dir) if n.finalized and since <= n.night_of <= until]


# -- in-progress sidecar -----------------------------------------------------


def save_current(sleep_dir: Path, night: SleepNight) -> None:
    """Mirror the open night to ``_current.json`` (atomic temp+rename)."""
    sleep_dir.mkdir(parents=True, exist_ok=True)
    path = sleep_dir / CURRENT_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_json(night), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_current(sleep_dir: Path) -> SleepNight | None:
    """Restore the open night, or None if absent/corrupt (never raises)."""
    path = sleep_dir / CURRENT_FILENAME
    try:
        return from_json(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except Exception:
        LOGGER.warning("Could not read in-progress sleep sidecar %s; ignoring", path)
        return None


def clear_current(sleep_dir: Path) -> None:
    """Remove the in-progress sidecar (at finalize)."""
    (sleep_dir / CURRENT_FILENAME).unlink(missing_ok=True)
