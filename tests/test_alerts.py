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


def test_two_models_same_object_alerts_once(monkeypatch) -> None:
    # Two models report the same perched bird every frame with slightly different
    # boxes. The second box sits inside the movement threshold of the first, so
    # only one alert fires and repeats never do (until the window lapses).
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=900.0, bbox_movement_alert_ratio=0.10)

    def frame():
        # model A box and model B box, ~2px apart on a 1000px frame.
        return state.eligible(
            "camera-1",
            [detection("bird", (100, 100, 140, 140)), detection("bird", (102, 101, 142, 141))],
            (1000, 1000),
        )

    assert labels(frame()) == ["bird"]
    now = 101.0
    assert frame() == []
    now = 102.0
    assert frame() == []


def test_two_models_divergent_boxes_do_not_alert_every_frame(monkeypatch) -> None:
    # The regression that caused the 429s: two models disagree on the box by more
    # than the movement threshold and detect intermittently, so the tracked
    # centre oscillated between them and fired on (almost) every frame. Anchoring
    # on alerted positions caps it at one alert per distinct position.
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=900.0, bbox_movement_alert_ratio=0.10)

    # Model A's box (centre x=20) and model B's box (centre x=60): 40% apart.
    box_a = (10, 10, 30, 30)
    box_b = (50, 10, 70, 30)

    def see(box):
        return state.eligible("camera-1", [detection("bird", box)], (100, 100))

    # First sighting of each distinct position alerts once...
    assert labels(see(box_a)) == ["bird"]
    now = 101.0
    assert labels(see(box_b)) == ["bird"]

    # ...then the oscillation produces no further alerts.
    for tick in range(102, 112):
        now = float(tick)
        assert see(box_a if tick % 2 == 0 else box_b) == []


def test_movement_accumulates_against_last_alert(monkeypatch) -> None:
    # Movement is measured from the last *alerted* position, not the last seen
    # one, so a slow drift that never trips frame-to-frame still alerts once its
    # cumulative displacement crosses the threshold.
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=900.0, bbox_movement_alert_ratio=0.10)

    assert labels(state.eligible("camera-1", [detection(bbox=(0, 0, 20, 20))], (100, 100))) == [
        "bird"
    ]  # anchor centre x=10
    now = 101.0
    # +6px: 0.06 < 0.10, no alert, anchor stays at x=10.
    assert state.eligible("camera-1", [detection(bbox=(6, 0, 26, 20))], (100, 100)) == []
    now = 102.0
    # +12px from the anchor: 0.12 >= 0.10, alert.
    assert labels(state.eligible("camera-1", [detection(bbox=(12, 0, 32, 20))], (100, 100))) == [
        "bird"
    ]


def test_reconfirms_after_window_lapses(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=900.0, bbox_movement_alert_ratio=0.10)

    assert labels(state.eligible("camera-1", [detection()], (100, 100))) == ["bird"]
    now = 500.0  # still inside the window, same spot: silent
    assert state.eligible("camera-1", [detection()], (100, 100)) == []
    now = 1001.0  # window lapsed: re-confirm the still-present bird once
    assert labels(state.eligible("camera-1", [detection()], (100, 100))) == ["bird"]


def test_label_case_folds_across_models(monkeypatch) -> None:
    # Two models labelling the same object "Bird" / "bird" share one alert
    # stream rather than each getting its own throttle.
    now = 100.0
    monkeypatch.setattr("lib.alerts.time.monotonic", lambda: now)
    state = AlertState(last_seen_alert_seconds=900.0, bbox_movement_alert_ratio=0.10)

    eligible = state.eligible(
        "camera-1",
        [detection("Bird", (0, 0, 20, 20)), detection("bird", (2, 1, 22, 21))],
        (1000, 1000),
    )
    assert len(eligible) == 1
