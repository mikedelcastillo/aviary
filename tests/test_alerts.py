from __future__ import annotations

from lib.alerts import AlertState
from lib.detector import Detection


def detection(label: str = "bird", bbox=(0, 0, 20, 20)) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=bbox)


def labels(detections: list[Detection]) -> list[str]:
    return [item.label for item in detections]


def test_first_sighting_alerts() -> None:
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    eligible = state.eligible("camera-1", [detection()], (100, 100))

    assert labels(eligible) == ["bird"]


def test_recent_stationary_sighting_does_not_alert(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    assert state.eligible("camera-1", [detection(bbox=(0, 0, 20, 20))], (100, 100))
    now = 110.0

    assert state.eligible("camera-1", [detection(bbox=(1, 1, 21, 21))], (100, 100)) == []


def test_last_alert_over_threshold_alerts_again(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    assert state.eligible("camera-1", [detection()], (100, 100))
    now = 110.0
    assert state.eligible("camera-1", [detection()], (100, 100)) == []
    now = 115.1

    assert labels(state.eligible("camera-1", [detection()], (100, 100))) == ["bird"]


def test_movement_at_threshold_alerts(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    assert state.eligible("camera-1", [detection(bbox=(0, 0, 20, 20))], (100, 100))
    now = 101.0

    assert labels(state.eligible("camera-1", [detection(bbox=(10, 0, 30, 20))], (100, 100))) == [
        "bird"
    ]


def test_movement_below_threshold_does_not_alert(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    assert state.eligible("camera-1", [detection(bbox=(0, 0, 20, 20))], (100, 100))
    now = 101.0

    assert state.eligible("camera-1", [detection(bbox=(9, 0, 29, 20))], (100, 100)) == []


def test_same_label_on_different_cameras_is_independent(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    assert state.eligible("camera-1", [detection()], (100, 100))
    now = 101.0

    assert labels(state.eligible("camera-2", [detection()], (100, 100))) == ["bird"]


def test_dedupes_label_per_frame() -> None:
    state = AlertState(last_seen_alert_seconds=15.0, bbox_movement_alert_ratio=0.10)

    eligible = state.eligible(
        "camera-1",
        [detection("bird", (0, 0, 20, 20)), detection("bird", (50, 50, 70, 70))],
        (100, 100),
    )

    assert labels(eligible) == ["bird"]
