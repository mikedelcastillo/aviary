"""Object sighting registry shared across cameras."""

from __future__ import annotations

import threading
import time


class ObjectRegistry:
    """Global tally of every label seen across all cameras.

    Shared by every camera's stats; the dashboard reads it to render the
    "objects" section. Each label tracks when it was last seen, how many frames
    it has appeared in, and which cameras saw it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, dict] = {}

    def record(self, labels: list[str], camera_name: str) -> None:
        now = time.monotonic()
        with self._lock:
            for label in labels:
                entry = self._objects.get(label)
                if entry is None:
                    entry = {"count": 0, "cameras": set()}
                    self._objects[label] = entry
                entry["last_seen"] = now
                entry["count"] += 1
                entry["cameras"].add(camera_name)

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            rows = [
                {
                    "label": label,
                    "since": now - entry["last_seen"],
                    "count": entry["count"],
                    "cameras": sorted(entry["cameras"]),
                }
                for label, entry in self._objects.items()
            ]
        # Smallest "since" (most recently seen) first.
        rows.sort(key=lambda row: row["since"])
        return rows
