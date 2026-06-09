"""Unit tests for aviary_immich.records — pure domain logic."""

from __future__ import annotations

import pytest

from aviary_immich.records import (
    _is_cuda_oom,
    asset_name,
    bump_decision,
    category_confidence,
    chunks,
    detected_labels,
    record_from_error,
    record_from_prediction,
    scan_postfix,
)
from aviary_immich.records import (
    aggregate_video_outputs,
    record_from_outputs,
)
from fakes import (
    OutOfMemoryError,
    make_asset,
    make_detection,
    make_output,
    make_prediction,
    make_record,
    make_tag,
)

KNOWN = {"bird", "dog", "cat"}


# --------------------------------------------------------------------------- detected_labels


def test_detected_labels_reads_labels_field():
    record = make_record(labels=["bird", "dog"])
    assert detected_labels(record, KNOWN) == {"bird", "dog"}


def test_detected_labels_lowercases_labels_field():
    record = make_record(labels=["BIRD", "Dog"])
    assert detected_labels(record, KNOWN) == {"bird", "dog"}


def test_detected_labels_intersects_with_known():
    record = make_record(labels=["bird", "horse"])
    assert detected_labels(record, KNOWN) == {"bird"}


def test_detected_labels_drops_unknown_only():
    record = make_record(labels=["horse", "sheep"])
    assert detected_labels(record, KNOWN) == set()


def test_detected_labels_drops_empty_and_none():
    record = make_record(labels=["bird", "", None])
    assert detected_labels(record, KNOWN) == {"bird"}


def test_detected_labels_falls_back_to_detections_when_labels_absent():
    record = make_record(
        labels=None,
        detections=[make_detection("bird"), make_detection("dog")],
    )
    assert detected_labels(record, KNOWN) == {"bird", "dog"}


def test_detected_labels_fallback_lowercases_detection_labels():
    record = make_record(labels=None, detections=[make_detection("BIRD")])
    assert detected_labels(record, KNOWN) == {"bird"}


def test_detected_labels_fallback_intersects_with_known():
    record = make_record(
        labels=None,
        detections=[make_detection("bird"), make_detection("horse")],
    )
    assert detected_labels(record, KNOWN) == {"bird"}


def test_detected_labels_fallback_drops_none_detection_label():
    record = make_record(labels=None, detections=[{"label": None}, make_detection("dog")])
    assert detected_labels(record, KNOWN) == {"dog"}


def test_detected_labels_no_labels_no_detections_is_empty():
    record = make_record(labels=None)
    assert detected_labels(record, KNOWN) == set()


def test_detected_labels_empty_labels_list_is_empty():
    record = make_record(labels=[])
    assert detected_labels(record, KNOWN) == set()


# --------------------------------------------------------------------------- category_confidence


def test_category_confidence_max_among_matching_detections():
    record = make_record(
        labels=None,
        detections=[
            make_detection("bird", confidence=0.4),
            make_detection("bird", confidence=0.7),
            make_detection("dog", confidence=0.9),
        ],
    )
    assert category_confidence(record, "bird") == pytest.approx(0.7)


def test_category_confidence_case_insensitive_match():
    record = make_record(labels=None, detections=[make_detection("BIRD", confidence=0.55)])
    assert category_confidence(record, "bird") == pytest.approx(0.55)


def test_category_confidence_falls_back_to_max_confidence_when_no_detection_matches():
    record = make_record(
        labels=None,
        detections=[make_detection("dog", confidence=0.8)],
        max_confidence=0.33,
    )
    assert category_confidence(record, "bird") == pytest.approx(0.33)


def test_category_confidence_falls_back_when_no_detections():
    record = make_record(labels=None, max_confidence=0.42)
    assert category_confidence(record, "bird") == pytest.approx(0.42)


def test_category_confidence_zero_when_neither():
    record = make_record(labels=None, max_confidence=0.0)
    assert category_confidence(record, "bird") == 0.0


def test_category_confidence_tolerates_none_confidence():
    record = make_record(
        labels=None,
        detections=[{"label": "bird", "confidence": None}],
        max_confidence=0.0,
    )
    assert category_confidence(record, "bird") == 0.0


def test_category_confidence_none_confidence_falls_through_to_other_match():
    record = make_record(
        labels=None,
        detections=[
            {"label": "bird", "confidence": None},
            make_detection("bird", confidence=0.6),
        ],
    )
    assert category_confidence(record, "bird") == pytest.approx(0.6)


# --------------------------------------------------------------------------- bump_decision


def test_bump_decision_error_increments_only_errors():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(decision="error"))
    assert stats == {"errors": 1}


def test_bump_decision_error_does_not_count_labels():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(decision="error", labels=["bird"]))
    assert stats == {"errors": 1}


def test_bump_decision_increments_per_matched_label():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(labels=["bird", "dog"]))
    assert stats == {"Birds": 1, "Dogs": 1}


def test_bump_decision_cat_label():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(labels=["cat"]))
    assert stats == {"Cats": 1}


def test_bump_decision_no_match_increments_other():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(labels=["horse"]))
    assert stats == {"other": 1}


def test_bump_decision_empty_labels_increments_other():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(labels=[]))
    assert stats == {"other": 1}


def test_bump_decision_accumulates_across_calls():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(labels=["bird"]))
    bump_decision(stats, make_record(labels=["bird"]))
    bump_decision(stats, make_record(decision="error"))
    assert stats == {"Birds": 2, "errors": 1}


def test_bump_decision_uses_detections_fallback():
    stats: dict[str, int] = {}
    bump_decision(stats, make_record(labels=None, detections=[make_detection("dog")]))
    assert stats == {"Dogs": 1}


# --------------------------------------------------------------------------- chunks


def test_chunks_exact_split():
    assert [list(c) for c in chunks([1, 2, 3, 4], 2)] == [[1, 2], [3, 4]]


def test_chunks_remainder_tail():
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunks_size_larger_than_items():
    assert list(chunks([1, 2], 10)) == [[1, 2]]


def test_chunks_size_zero_coerced_to_one():
    assert list(chunks([1, 2, 3], 0)) == [[1], [2], [3]]


def test_chunks_negative_size_coerced_to_one():
    assert list(chunks([1, 2], -5)) == [[1], [2]]


def test_chunks_empty_list_yields_nothing():
    assert list(chunks([], 3)) == []


# --------------------------------------------------------------------------- scan_postfix


def test_scan_postfix_full_stats():
    stats = {"Birds": 1, "Dogs": 2, "Cats": 3, "Tennis": 7, "errors": 4}
    assert scan_postfix(stats) == "Birds=1 Dogs=2 Cats=3 Tennis=7 err=4"


def test_scan_postfix_missing_keys_default_zero():
    assert scan_postfix({}) == "Birds=0 Dogs=0 Cats=0 Tennis=0 err=0"


def test_scan_postfix_prepends_prefix():
    assert scan_postfix({}, prefix="scan ") == "scan Birds=0 Dogs=0 Cats=0 Tennis=0 err=0"


def test_scan_postfix_partial_stats():
    assert scan_postfix({"Birds": 5}) == "Birds=5 Dogs=0 Cats=0 Tennis=0 err=0"


# --------------------------------------------------------------------------- record_from_prediction


def test_record_from_prediction_match_with_detections(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "FIXED-TS")
    asset = make_asset(id="a7", name="photo.jpg")
    prediction = make_prediction(labels=["bird"], confidence=0.8)
    record = record_from_prediction(asset, "acct", prediction)
    assert record["decision"] == "match"
    assert record["labels"] == ["bird"]
    assert record["max_confidence"] == pytest.approx(0.8)
    assert record["account"] == "acct"
    assert record["asset_id"] == "a7"
    assert record["original_file_name"] == "photo.jpg"
    assert record["scanned_at"] == "FIXED-TS"
    assert record["model_tags"] == {"yolo": {"bird": pytest.approx(0.8)}}
    assert len(record["detections"]) == 1
    detection = record["detections"][0]
    assert detection["label"] == "bird"
    assert detection["confidence"] == pytest.approx(0.8)
    assert detection["model"] == "yolo"


def test_record_from_prediction_not_match_without_detections(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "FIXED-TS")
    asset = make_asset(id="a1")
    prediction = make_prediction(detections=[])
    record = record_from_prediction(asset, "acct", prediction)
    assert record["decision"] == "not_match"
    assert record["labels"] == []


def test_record_from_prediction_labels_sorted_unique_lowercased(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id="a1")
    detections = [
        make_detection("Dog"),
        make_detection("bird"),
        make_detection("BIRD"),
    ]
    prediction = make_prediction(detections=detections)
    record = record_from_prediction(asset, "acct", prediction)
    assert record["labels"] == ["bird", "dog"]


def test_record_from_prediction_asset_id_coerced_to_str(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id=12345)
    prediction = make_prediction(detections=[])
    record = record_from_prediction(asset, "acct", prediction)
    assert record["asset_id"] == "12345"


def test_record_from_prediction_uses_id_fallback_for_name(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id="a9")
    prediction = make_prediction(detections=[])
    record = record_from_prediction(asset, "acct", prediction)
    assert record["original_file_name"] == "a9"


# --------------------------------------------------------------------------- record_from_outputs


def test_record_from_outputs_merges_two_models(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id="a1", name="photo.jpg")
    outputs = {
        "yolo": make_output(tags=[make_tag("bird", 0.8)], max_confidence=0.8),
        "clip": make_output(tags=[make_tag("cat", 0.6, box=None)], max_confidence=0.6),
    }
    record = record_from_outputs(asset, "acct", outputs)
    assert record["model_tags"] == {
        "yolo": {"bird": pytest.approx(0.8)},
        "clip": {"cat": pytest.approx(0.6)},
    }
    assert record["labels"] == ["bird", "cat"]
    assert record["decision"] == "match"
    assert record["max_confidence"] == pytest.approx(0.8)
    # The scene tag (box=None) contributes to labels/model_tags but not detections.
    assert len(record["detections"]) == 1
    assert record["detections"][0]["model"] == "yolo"
    assert record["detections"][0]["label"] == "bird"


# --------------------------------------------------------------------------- aggregate_video_outputs


def test_aggregate_video_outputs_keeps_best_tag_across_frames(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id="vid1", name="clip.mp4")
    frame_outputs = [
        {"yolo": make_output(tags=[make_tag("bird", 0.4)], max_confidence=0.4)},
        {"yolo": make_output(tags=[make_tag("bird", 0.9)], max_confidence=0.9)},
        {"yolo": make_output(tags=[make_tag("bird", 0.7)], max_confidence=0.7)},
    ]
    record = aggregate_video_outputs(asset, "acct", frame_outputs, frames_scanned=3)
    assert record["asset_type"] == "video"
    assert record["frames_scanned"] == 3
    assert record["labels"] == ["bird"]
    assert record["model_tags"] == {"yolo": {"bird": pytest.approx(0.9)}}
    assert record["max_confidence"] == pytest.approx(0.9)
    assert len(record["detections"]) == 1
    assert record["detections"][0]["confidence"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- record_from_error


def test_record_from_error_decision_and_message(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "ERR-TS")
    asset = make_asset(id="a3", name="bad.jpg")
    record = record_from_error(asset, "acct", ValueError("boom"))
    assert record["decision"] == "error"
    assert record["error"] == "boom"
    assert record["account"] == "acct"
    assert record["asset_id"] == "a3"
    assert record["original_file_name"] == "bad.jpg"
    assert record["scanned_at"] == "ERR-TS"


def test_record_from_error_asset_id_coerced(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    record = record_from_error(make_asset(id=99), "acct", RuntimeError("x"))
    assert record["asset_id"] == "99"


# --------------------------------------------------------------------------- asset_name


def test_asset_name_prefers_original_file_name():
    asset = make_asset(id="a1", name="file.jpg", path="/some/path.jpg")
    assert asset_name(asset) == "file.jpg"


def test_asset_name_falls_back_to_original_path():
    asset = make_asset(id="a1", path="/some/path.jpg")
    assert asset_name(asset) == "/some/path.jpg"


def test_asset_name_falls_back_to_id():
    asset = make_asset(id="a1")
    assert asset_name(asset) == "a1"


def test_asset_name_empty_fallback():
    assert asset_name({}) == ""


def test_asset_name_empty_string_id_falls_to_empty():
    assert asset_name({"id": ""}) == ""


# --------------------------------------------------------------------------- _is_cuda_oom


def test_is_cuda_oom_true_for_fakes_class():
    assert _is_cuda_oom(OutOfMemoryError("CUDA out of memory")) is True


def test_is_cuda_oom_true_for_fakes_class_empty_message():
    assert _is_cuda_oom(OutOfMemoryError()) is True


def test_is_cuda_oom_true_for_message_match():
    assert _is_cuda_oom(RuntimeError("CUDA out of memory")) is True


def test_is_cuda_oom_message_match_case_insensitive():
    assert _is_cuda_oom(RuntimeError("Out Of Memory")) is True


def test_is_cuda_oom_false_for_unrelated_exception():
    assert _is_cuda_oom(ValueError("something else")) is False
