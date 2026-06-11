from __future__ import annotations

import json

import numpy as np

from lib.collect import collect_alerted_detections, safe_path_component
from lib.config import CameraConfig, CollectConfig
from lib.detector import Detection


def test_safe_path_component_removes_path_separators() -> None:
    assert safe_path_component("../bird feeder") == "___bird_feeder"


def test_collect_alerted_detections_writes_raw_image_and_metadata(tmp_path) -> None:
    config = CollectConfig(objects=frozenset({"bird"}), directory=tmp_path)
    camera = CameraConfig(name="camera-1", enabled=True, rtsp_url="rtsp://example")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    detection = Detection(label="bird", confidence=0.91, bbox_xyxy=(1, 2, 11, 12))

    collected = collect_alerted_detections(config, camera, frame, [detection])

    assert len(collected) == 1
    image_path, metadata_path = collected[0]
    assert image_path.parent == tmp_path / "bird"
    assert image_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["object"] == "bird"
    assert metadata["camera"]["name"] == "camera-1"
    assert metadata["frame"] == {"width": 30, "height": 20}
    assert metadata["detection"]["bbox_xyxy"] == {"x1": 1, "y1": 2, "x2": 11, "y2": 12}
    assert metadata["detection"]["center"] == {"x": 6.0, "y": 7.0}


def test_collect_alerted_detections_ignores_unconfigured_objects(tmp_path) -> None:
    config = CollectConfig(objects=frozenset({"bird"}), directory=tmp_path)
    camera = CameraConfig(name="camera-1", enabled=True, rtsp_url="rtsp://example")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    detection = Detection(label="person", confidence=0.91, bbox_xyxy=(1, 2, 11, 12))

    assert collect_alerted_detections(config, camera, frame, [detection]) == []
    assert not list(tmp_path.iterdir())
