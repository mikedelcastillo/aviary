"""Daily JSON detection activity logs.

Each positive inference updates a compact per-day JSON file under
``data/server/detection``. The file stores merged time intervals per
camera+label plus counts/confidence, so a day can answer questions like "how long
was Percy visible on the food camera?" without keeping every sampled frame.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from lib.clock import PH_TZ
from lib.detector import Detection


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _day_path(directory: Path, observed_at: datetime) -> Path:
    # Key each file by the flock's LOCAL (Manila) calendar day, not UTC, so "today"
    # means the same 24h window here as it does for /activity, /sleep and the
    # caretaker. The timestamps stored inside stay UTC; only the day boundary is
    # local. (UTC keying rolled the day at 08:00 Manila — mid-morning — which made
    # /detections and /status disagree with everything else.)
    return directory / f"{observed_at.astimezone(PH_TZ).date().isoformat()}.json"


@dataclass(frozen=True)
class DetectionActivity:
    camera: str
    label: str
    total_seconds: float
    observations: int
    intervals: list[dict]
    last_seen_at: datetime | None = None


class DetectionLogger:
    """Append-like daily detection logger with merged presence intervals."""

    def __init__(
        self,
        directory: Path = Path("./data/server/detection"),
        *,
        merge_gap_seconds: float = 3.0,
        flush_interval_seconds: float = 0.0,
    ) -> None:
        self._directory = directory
        self._merge_gap_seconds = merge_gap_seconds
        self._lock = threading.Lock()
        # The active day's state lives in memory; the file is a periodic
        # snapshot. At 0 (the default) every record() still writes through
        # immediately — the server passes a real interval so busy hours don't
        # re-serialize a ~1MB day file on every detection-positive frame from
        # every camera thread. Call flush() at shutdown to persist the tail.
        self._flush_interval_seconds = flush_interval_seconds
        self._cache_path: Path | None = None
        self._cache: dict | None = None
        self._dirty = False
        self._last_flush = 0.0

    def record(
        self,
        *,
        camera_name: str,
        detections: list[Detection],
        frame_size: tuple[int, int],
        sample_interval_seconds: float,
        observed_at: datetime | None = None,
    ) -> None:
        if not detections:
            return
        observed_at = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        duration = max(0.0, float(sample_interval_seconds))
        end_at = observed_at.timestamp() + duration

        with self._lock:
            path = _day_path(self._directory, observed_at)
            data = self._load_day_locked(path, observed_at)
            camera = data["cameras"].setdefault(camera_name, {"labels": {}})
            labels = camera.setdefault("labels", {})
            width, height = frame_size

            # Multiple boxes with the same label in one frame count as one
            # visibility observation for duration, but retain max confidence.
            by_label: dict[str, list[Detection]] = {}
            for detection in detections:
                by_label.setdefault(detection.label, []).append(detection)

            for label, label_detections in by_label.items():
                entry = labels.setdefault(
                    label,
                    {
                        "total_detected_seconds": 0.0,
                        "observations": 0,
                        "max_confidence": 0.0,
                        "last_seen_at": None,
                        "frame": {"width": width, "height": height},
                        "intervals": [],
                    },
                )
                entry["observations"] += 1
                entry["max_confidence"] = max(
                    float(entry.get("max_confidence") or 0.0),
                    max(float(d.confidence) for d in label_detections),
                )
                entry["last_seen_at"] = _iso(observed_at)
                entry["frame"] = {"width": width, "height": height}
                self._merge_interval(entry["intervals"], observed_at.timestamp(), end_at)
                entry["total_detected_seconds"] = round(
                    sum(float(item["duration_seconds"]) for item in entry["intervals"]), 3
                )

            data["updated_at"] = _iso(datetime.now(timezone.utc))
            self._dirty = True
            if (
                self._flush_interval_seconds <= 0
                or time.monotonic() - self._last_flush >= self._flush_interval_seconds
            ):
                self._flush_locked()

    def flush(self) -> None:
        """Persist any buffered day state now (shutdown / rollover hook)."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._dirty and self._cache is not None and self._cache_path is not None:
            self._write_day(self._cache_path, self._cache)
        self._dirty = False
        self._last_flush = time.monotonic()

    def _load_day_locked(self, path: Path, observed_at: datetime) -> dict:
        if self._cache_path == path and self._cache is not None:
            return self._cache
        # Day rolled over (or first use): persist the outgoing day before the
        # cache moves on, so its tail is never lost to the new day's buffer.
        self._flush_locked()
        self._cache_path = path
        self._cache = self._read_day(path, observed_at)
        return self._cache

    def activity_for_day(
        self,
        day: datetime,
        *,
        label: str | None = None,
        camera_name: str | None = None,
    ) -> list[DetectionActivity]:
        path = _day_path(self._directory, day)
        rows: list[DetectionActivity] = []
        with self._lock:
            # The active day answers straight from memory (fresher than disk
            # between flushes, and skips a ~1MB parse per query); other days
            # come from their files. Rows copy the interval dicts so callers
            # never alias state that record() keeps mutating.
            if self._cache_path == path and self._cache is not None:
                data = self._cache
            else:
                data = self._read_day(path, day) if path.exists() else None
            if data is None:
                return []
            for camera, camera_data in data.get("cameras", {}).items():
                if camera_name is not None and camera != camera_name:
                    continue
                for item_label, entry in camera_data.get("labels", {}).items():
                    if label is not None and item_label != label:
                        continue
                    rows.append(
                        DetectionActivity(
                            camera=camera,
                            label=item_label,
                            total_seconds=float(entry.get("total_detected_seconds") or 0.0),
                            observations=int(entry.get("observations") or 0),
                            intervals=[dict(item) for item in entry.get("intervals") or []],
                            last_seen_at=(
                                _parse_iso(entry["last_seen_at"])
                                if entry.get("last_seen_at")
                                else None
                            ),
                        )
                    )
        rows.sort(key=lambda row: (row.label, row.camera))
        return rows

    def _merge_interval(self, intervals: list[dict], start_ts: float, end_ts: float) -> None:
        if intervals:
            last = intervals[-1]
            last_end = _parse_iso(last["end_at"]).timestamp()
            if start_ts <= last_end + self._merge_gap_seconds:
                merged_end = max(last_end, end_ts)
                last["end_at"] = _iso(datetime.fromtimestamp(merged_end, timezone.utc))
                last["duration_seconds"] = round(
                    merged_end - _parse_iso(last["start_at"]).timestamp(), 3
                )
                return
        intervals.append(
            {
                "start_at": _iso(datetime.fromtimestamp(start_ts, timezone.utc)),
                "end_at": _iso(datetime.fromtimestamp(end_ts, timezone.utc)),
                "duration_seconds": round(end_ts - start_ts, 3),
            }
        )

    def _read_day(self, path: Path, observed_at: datetime) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "date": observed_at.astimezone(PH_TZ).date().isoformat(),
            "timezone": "Asia/Manila",
            "created_at": _iso(datetime.now(timezone.utc)),
            "updated_at": _iso(datetime.now(timezone.utc)),
            "cameras": {},
        }

    def _write_day(self, path: Path, data: dict) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._directory,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(path)
