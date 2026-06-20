from __future__ import annotations

import json
from pathlib import Path

from aviary_training.annotations import is_training_ready, review_status


def _image(tmp_path: Path, stem: str) -> Path:
    """An image file; its bytes are irrelevant — only the sibling sidecars matter."""
    image = tmp_path / f"{stem}.jpg"
    image.write_bytes(b"")
    return image


def _write_json(image: Path, payload: dict) -> None:
    image.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")


def test_unreviewed_placeholder_txt_is_not_training_ready(tmp_path: Path) -> None:
    # Auto-collected frame: a YOLO placeholder .txt (class 0) but NO .json sidecar.
    image = _image(tmp_path, "camera-2_day_2026-06-12_014400_ca5b36bb")
    image.with_suffix(".txt").write_text("0 0.4 0.7 0.1 0.4\n", encoding="utf-8")

    assert review_status(image) == "unreviewed"
    assert is_training_ready(image) is False


def test_reviewed_and_labeled_is_training_ready(tmp_path: Path) -> None:
    image = _image(tmp_path, "reviewed")
    _write_json(
        image,
        {"boxed": True, "boxes": [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "label": "percy"}]},
    )

    assert review_status(image) == "ready"
    assert is_training_ready(image) is True


def test_reviewed_empty_negative_is_training_ready(tmp_path: Path) -> None:
    # boxed=true with no boxes is a human-confirmed "nothing here" background frame.
    image = _image(tmp_path, "negative")
    _write_json(image, {"boxed": True, "boxes": []})

    assert review_status(image) == "ready"
    assert is_training_ready(image) is True


def test_boxed_but_unlabeled_box_is_not_training_ready(tmp_path: Path) -> None:
    # Boxed but still has an unlabeled box: in the label queue, not done. Training it
    # would teach the model that the un-exported bird is background.
    image = _image(tmp_path, "in_progress")
    _write_json(
        image,
        {"boxed": True, "boxes": [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "label": None}]},
    )

    assert review_status(image) == "incomplete"
    assert is_training_ready(image) is False


def test_seeded_but_not_boxed_is_not_training_ready(tmp_path: Path) -> None:
    # A .json with boxed=false (opened/seeded but not human-confirmed) is not ready.
    image = _image(tmp_path, "seeded")
    _write_json(
        image,
        {"boxed": False, "boxes": [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "label": "percy"}]},
    )

    assert review_status(image) == "unboxed"
    assert is_training_ready(image) is False


def test_corrupt_json_is_treated_as_unreviewed(tmp_path: Path) -> None:
    image = _image(tmp_path, "corrupt")
    image.with_suffix(".json").write_text("{ not json", encoding="utf-8")

    assert review_status(image) == "unreviewed"
    assert is_training_ready(image) is False
