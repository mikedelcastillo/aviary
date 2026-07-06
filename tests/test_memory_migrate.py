from __future__ import annotations

from lib.memory_migrate import _needs_migration, _raw_to_observation


def test_raw_to_observation_preserves_stored_detections_and_provenance() -> None:
    # An outage-era v3 record carries YOLO identity (detections + boxes) but no
    # VLM decoration. When migrate's CPU re-detection misses on its photo, the
    # preserved observation must keep those stored detections — dropping them
    # would destroy identity data an outage already survived once.
    raw = {
        "camera": "Big Cage",
        "birds": ["percy"],
        "note": "",
        "photo": "data/x.jpg",
        "detections": [
            {"label": "percy", "confidence": 0.91, "bbox": [1, 2, 3, 4], "activity": "eating"}
        ],
        "detector_model": "live-019.pt",
    }
    obs = _raw_to_observation(raw)
    assert len(obs.detections) == 1
    det = obs.detections[0]
    assert (det.label, det.confidence, tuple(det.bbox)) == ("percy", 0.91, (1, 2, 3, 4))
    assert det.activity == "eating"
    assert obs.detector_model == "live-019.pt"
    assert obs.vlm_model == ""  # still undecorated — a later run can finish it


def test_outage_era_v3_record_needs_migration() -> None:
    # The provenance fix leaves vlm_model EMPTY when the VLM never ran; those
    # records must be picked up by a normal (non --force) migrate run.
    raw = {
        "version": 3,
        "observations": [{"photo": "x.jpg", "detections": [{"label": "percy"}]}],
    }
    assert _needs_migration(raw, force=False) is True
    raw["observations"][0]["vlm_model"] = "qwen2.5vl:7b"
    assert _needs_migration(raw, force=False) is False
