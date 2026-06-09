from __future__ import annotations

from lib.detector import Detection
from lib.objects import ObjectRegistry


def detection(label: str = "bird", bbox=(0, 0, 20, 20)) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=bbox)


def test_registry_rows_are_camera_specific() -> None:
    registry = ObjectRegistry()

    registry.record([detection()], "camera-1", (100, 100))
    registry.record([detection()], "camera-2", (100, 100))

    rows = registry.snapshot()
    assert {(row["camera"], row["label"]) for row in rows} == {
        ("camera-1", "bird"),
        ("camera-2", "bird"),
    }


def test_registry_calculates_movement_percent_from_previous_center() -> None:
    registry = ObjectRegistry()

    registry.record([detection(bbox=(0, 0, 20, 20))], "camera-1", (100, 100))
    registry.record([detection(bbox=(10, 5, 30, 25))], "camera-1", (100, 100))

    [row] = registry.snapshot()
    assert row["previous_center"] == (10.0, 10.0)
    assert row["center"] == (20.0, 15.0)
    assert row["movement_percent"] == 10.0


def test_registry_first_sighting_has_no_movement_percent() -> None:
    registry = ObjectRegistry()

    registry.record([detection()], "camera-1", (100, 100))

    [row] = registry.snapshot()
    assert row["movement_percent"] is None
    assert row["since_alert"] is None


def test_registry_tracks_last_alert(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.objects.time.monotonic", lambda: now)
    registry = ObjectRegistry()

    bird = detection()
    registry.record([bird], "camera-1", (100, 100))
    now = 103.0
    registry.record_alert([bird], "camera-1")
    now = 108.0

    [row] = registry.snapshot()
    assert row["since_alert"] == 5.0


def test_registry_sorts_most_recent_first(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.objects.time.monotonic", lambda: now)
    registry = ObjectRegistry()

    registry.record([detection("older")], "camera-1", (100, 100))
    now = 105.0
    registry.record([detection("newer")], "camera-1", (100, 100))

    assert [row["label"] for row in registry.snapshot()] == ["newer", "older"]
