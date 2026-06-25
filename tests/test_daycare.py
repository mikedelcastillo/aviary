from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from lib.daycare import QUIET_REASSURE_AFTER, DaycareNarrator


def _write_sighting(collect_dir, label, conf, when_epoch) -> None:
    folder = collect_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{int(when_epoch)}"
    (folder / f"{stem}.jpg").write_bytes(b"\xff\xd8jpeg")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "object": label,
                "camera": {"name": "camera-192.168.1.8"},
                "collected_at": datetime.fromtimestamp(when_epoch, timezone.utc).isoformat(),
                "frame": {"width": 100, "height": 100},
                "detection": {"confidence": conf, "bbox_xyxy": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}},
            }
        )
    )


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.albums: list[list] = []

    def broadcast_text(self, text: str) -> None:
        self.texts.append(text)

    def broadcast_album(self, items) -> None:
        self.albums.append(list(items))


class FakeClient:
    def generate(self, model, prompt, *, images=None, timeout_seconds=None, **kwargs):
        return "doing bird things"

    def chat(self, model, messages, **kwargs):
        return "Percy and Matcha had a lovely afternoon together."


def _narrator(collect_dir, notifier):
    return DaycareNarrator(
        collect_dir,
        FakeClient(),
        "qwen3:4b",
        "qwen2.5vl:7b",
        notifier,
        threading.Event(),
        interval_seconds=1.0,
        camera_display=lambda name: "Big Cage",
    )


def test_run_digest_sends_summary_and_album(tmp_path) -> None:
    _write_sighting(tmp_path, "percy", 0.9, 1000)
    _write_sighting(tmp_path, "matcha", 0.8, 1001)
    notifier = FakeNotifier()
    sent = _narrator(tmp_path, notifier).run_digest(0, 2000)

    assert sent is True
    assert any("Daycare update" in t for t in notifier.texts)
    assert any("lovely afternoon" in t for t in notifier.texts)
    assert notifier.albums and len(notifier.albums[0]) == 2  # one photo per bird


def test_run_digest_quiet_reassures_after_streak(tmp_path) -> None:
    notifier = FakeNotifier()
    narrator = _narrator(tmp_path, notifier)
    # Empty windows: silent until the streak threshold, then one reassurance.
    for _ in range(QUIET_REASSURE_AFTER - 1):
        assert narrator.run_digest(0, 1) is False
    assert notifier.texts == []
    narrator.run_digest(0, 1)
    assert len(notifier.texts) == 1
    assert "quiet" in notifier.texts[0].lower()
