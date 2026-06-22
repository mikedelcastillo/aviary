"""Tests for aviary_training.suggest — the pure logic behind the model-assisted
annotation suggestion scripts (IoU, detection→box matching, sidecar IO, and the
latest-model scan). Imports only the helper module, never ultralytics/torch, so
it stays fast and isolated.
"""

from __future__ import annotations

import json

import pytest

from aviary_training.suggest import (
    dedupe_suggestions,
    empty_suggestions,
    iou,
    is_boxed,
    latest_model,
    match_box,
    read_annotation_boxes,
    read_suggestions,
    suggestion_path,
    write_suggestions,
)


def box(cx, cy, w, h, **extra):
    return {"cx": cx, "cy": cy, "w": w, "h": h, **extra}


# --- IoU --------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    b = box(0.5, 0.5, 0.2, 0.2)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou(box(0.1, 0.1, 0.1, 0.1), box(0.9, 0.9, 0.1, 0.1)) == 0.0


def test_iou_half_overlap():
    # Two 0.2-wide, full-height-0.2 boxes offset by half their width along x:
    # intersection 0.1x0.2, union = 2*0.04 - 0.02 = 0.06 -> 0.02/0.06 = 1/3.
    a = box(0.4, 0.5, 0.2, 0.2)
    b = box(0.5, 0.5, 0.2, 0.2)
    assert iou(a, b) == pytest.approx(1 / 3)


# --- match_box --------------------------------------------------------------


def test_match_box_picks_highest_overlap():
    detection = box(0.5, 0.5, 0.2, 0.2)
    candidates = [
        box(0.1, 0.1, 0.2, 0.2),   # far away
        box(0.52, 0.5, 0.2, 0.2),  # strong overlap
        box(0.5, 0.5, 0.2, 0.2),   # exact -> best
    ]
    assert match_box(detection, candidates, threshold=0.5) == 2


def test_match_box_below_threshold_returns_none():
    detection = box(0.5, 0.5, 0.2, 0.2)
    candidates = [box(0.62, 0.5, 0.2, 0.2)]  # only slight overlap
    assert match_box(detection, candidates, threshold=0.5) is None


def test_match_box_empty_candidates_returns_none():
    assert match_box(box(0.5, 0.5, 0.2, 0.2), [], threshold=0.5) is None


# --- dedupe_suggestions -----------------------------------------------------


def test_dedupe_drops_bird_overlapping_specific_box():
    # A generic 'bird' box and a specific 'draft' box on the same spot -> keep
    # only the specific one (bird is deprioritized).
    boxes = [
        box(0.5, 0.5, 0.2, 0.2, label="bird", conf=0.9),
        box(0.5, 0.5, 0.2, 0.2, label="draft", conf=0.4),
    ]
    out = dedupe_suggestions(boxes, iou_threshold=0.6)
    assert [b["label"] for b in out] == ["draft"]


def test_dedupe_drops_unlabeled_overlapping_specific_box():
    # label=None is treated as low priority just like 'bird'.
    boxes = [
        box(0.5, 0.5, 0.2, 0.2, label=None, conf=0.95),
        box(0.5, 0.5, 0.2, 0.2, label="matcha", conf=0.3),
    ]
    out = dedupe_suggestions(boxes, iou_threshold=0.6)
    assert [b["label"] for b in out] == ["matcha"]


def test_dedupe_two_custom_boxes_keeps_higher_conf():
    boxes = [
        box(0.5, 0.5, 0.2, 0.2, label="matcha", conf=0.4),
        box(0.5, 0.5, 0.2, 0.2, label="percy", conf=0.8),
    ]
    out = dedupe_suggestions(boxes, iou_threshold=0.6)
    assert [b["label"] for b in out] == ["percy"]


def test_dedupe_non_overlapping_all_kept():
    boxes = [
        box(0.1, 0.1, 0.1, 0.1, label="bird", conf=0.9),
        box(0.5, 0.5, 0.1, 0.1, label="draft", conf=0.5),
        box(0.9, 0.9, 0.1, 0.1, label=None, conf=0.7),
    ]
    out = dedupe_suggestions(boxes, iou_threshold=0.6)
    assert out == boxes  # order + identity preserved


def test_dedupe_keeps_bird_when_it_overlaps_nothing():
    boxes = [
        box(0.2, 0.2, 0.2, 0.2, label="percy", conf=0.8),
        box(0.8, 0.8, 0.2, 0.2, label="bird", conf=0.6),
    ]
    out = dedupe_suggestions(boxes, iou_threshold=0.6)
    assert [b["label"] for b in out] == ["percy", "bird"]


def test_dedupe_preserves_input_order():
    # Even though the higher-conf 'percy' is considered first internally, the
    # surviving boxes come back in original input order.
    boxes = [
        box(0.5, 0.5, 0.2, 0.2, label="percy", conf=0.3),
        box(0.1, 0.1, 0.1, 0.1, label="matcha", conf=0.9),
    ]
    out = dedupe_suggestions(boxes, iou_threshold=0.6)
    assert [b["label"] for b in out] == ["percy", "matcha"]


def test_dedupe_empty_returns_empty():
    assert dedupe_suggestions([]) == []


# --- Sidecar IO -------------------------------------------------------------


def test_suggestion_path_does_not_clobber_json(tmp_path):
    img = tmp_path / "camera-1_day_0001.jpg"
    assert suggestion_path(img).name == "camera-1_day_0001.suggest.json"
    assert img.with_suffix(".json").name == "camera-1_day_0001.json"


def test_read_suggestions_missing_is_empty(tmp_path):
    assert read_suggestions(tmp_path / "missing.jpg") == empty_suggestions()


def test_write_then_read_round_trips(tmp_path):
    img = tmp_path / "x.jpg"
    data = {
        "boxModel": "yolo11n.pt",
        "boxes": [box(0.5, 0.5, 0.1, 0.1, id="sug-0", conf=0.8)],
        "labels": [],
    }
    write_suggestions(img, data)
    assert read_suggestions(img) == data


def test_rewrite_preserves_unset_sidecar_fields(tmp_path):
    """read_suggestions -> mutate -> write_suggestions preserves fields the caller
    did not touch. suggest.py relies on this: it reads the existing sidecar before
    re-setting boxes/labels so any prior structure round-trips intact."""
    img = tmp_path / "x.jpg"
    write_suggestions(img, {"boxes": [], "labels": [{"boxId": "b1", "label": "percy", "conf": 0.6}]})

    cur = read_suggestions(img)
    cur["boxes"] = [box(0.5, 0.5, 0.1, 0.1, id="sug-0", conf=0.8)]
    cur["boxModel"] = "yolo11n.pt"
    write_suggestions(img, cur)

    after = read_suggestions(img)
    assert after["labels"] == [{"boxId": "b1", "label": "percy", "conf": 0.6}]
    assert len(after["boxes"]) == 1


# --- Annotation reads -------------------------------------------------------


def test_read_annotation_boxes_from_json(tmp_path):
    img = tmp_path / "x.jpg"
    (tmp_path / "x.json").write_text(
        json.dumps(
            {
                "boxed": True,
                "boxes": [
                    {"id": "b1", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "label": "pizza"},
                    {"id": "b2", "cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1, "label": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    boxes = read_annotation_boxes(img)
    assert [b["id"] for b in boxes] == ["b1", "b2"]
    assert boxes[0]["label"] == "pizza"
    assert is_boxed(img) is True


def test_read_annotation_boxes_missing_is_empty(tmp_path):
    img = tmp_path / "x.jpg"
    assert read_annotation_boxes(img) == []
    assert is_boxed(img) is False


# --- latest_model -----------------------------------------------------------


def test_latest_model_picks_highest_sequence(tmp_path):
    for name in ("live-001.pt", "live-002.pt", "live-010.pt", "archive-005.pt"):
        (tmp_path / name).write_bytes(b"")
    picked = latest_model(tmp_path, "live")
    assert picked is not None and picked.name == "live-010.pt"


def test_latest_model_none_when_absent(tmp_path):
    assert latest_model(tmp_path, "live") is None
