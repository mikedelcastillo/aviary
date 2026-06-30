from __future__ import annotations

import json
from datetime import datetime, timezone

from lib.detection_log import DetectionLogger
from lib.detector import Detection


def detection(label: str = "percy", confidence: float = 0.9) -> Detection:
    return Detection(label=label, confidence=confidence, bbox_xyxy=(10, 20, 30, 40))


def test_detection_logger_writes_daily_json_and_merges_intervals(tmp_path) -> None:
    logger = DetectionLogger(tmp_path, merge_gap_seconds=3.0)
    day = datetime(2026, 6, 27, 10, 0, 0, tzinfo=timezone.utc)

    logger.record(
        camera_name="camera-10.0.0.5",
        detections=[detection("percy", 0.7)],
        frame_size=(1920, 1080),
        sample_interval_seconds=1.0,
        observed_at=day,
    )
    logger.record(
        camera_name="camera-10.0.0.5",
        detections=[detection("percy", 0.95)],
        frame_size=(1920, 1080),
        sample_interval_seconds=1.0,
        observed_at=datetime(2026, 6, 27, 10, 0, 1, tzinfo=timezone.utc),
    )

    path = tmp_path / "2026-06-27.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["cameras"]["camera-10.0.0.5"]["labels"]["percy"]

    assert entry["observations"] == 2
    assert entry["max_confidence"] == 0.95
    assert entry["total_detected_seconds"] == 2.0
    assert entry["intervals"] == [
        {
            "start_at": "2026-06-27T10:00:00Z",
            "end_at": "2026-06-27T10:00:02Z",
            "duration_seconds": 2.0,
        }
    ]


def test_detection_logger_query_filters_day_label_and_camera(tmp_path) -> None:
    logger = DetectionLogger(tmp_path)
    day = datetime(2026, 6, 27, 10, 0, 0, tzinfo=timezone.utc)
    logger.record(
        camera_name="camera-1",
        detections=[detection("percy"), detection("matcha")],
        frame_size=(100, 100),
        sample_interval_seconds=2.0,
        observed_at=day,
    )
    logger.record(
        camera_name="camera-2",
        detections=[detection("percy")],
        frame_size=(100, 100),
        sample_interval_seconds=1.0,
        observed_at=day,
    )

    rows = logger.activity_for_day(day, label="percy", camera_name="camera-1")

    assert len(rows) == 1
    assert rows[0].camera == "camera-1"
    assert rows[0].label == "percy"
    assert rows[0].total_seconds == 2.0
    assert rows[0].observations == 1
    assert rows[0].last_seen_at == day
