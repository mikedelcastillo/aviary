"""Tests for training/scripts/import_collect_birds.py.

Covers the collect-as-SUGGESTION behavior: a collected detection becomes a
yellow ``<image>.suggest.json`` proposal (not a green ``.txt`` seed), and the
collection folder's name flows through as the roster identity.

The script is loaded by path (it imports rich/PIL and lives outside the package),
mirroring tests/test_prepare_dataset.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from aviary_training.suggest import read_suggestions, suggestion_path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "scripts"
    / "import_collect_birds.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("import_collect_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _detection(width=100, height=200, conf=0.8) -> dict:
    return {
        "frame": {"width": width, "height": height},
        "detection": {
            "confidence": conf,
            "bbox_xyxy": {"x1": 10, "y1": 20, "x2": 50, "y2": 120},
        },
    }


def _args(**overrides) -> SimpleNamespace:
    base = {"ir_threshold": 0.01, "no_labels": False, "dry_run": False, "workers": 1}
    base.update(overrides)
    return SimpleNamespace(**base)


def _color_jpg(path: Path) -> None:
    # A vivid color image -> classified as 'day' (high colored fraction).
    Image.new("RGB", (16, 16), (200, 30, 30)).save(path)


# --- collect_folders includes snapshots -------------------------------------


def test_snapshot_stem_is_matched_by_import_stem_regex():
    # snapshot_stem (server side) and STEM_RE (this importer) are independent
    # literals that must stay in lockstep, or snapshots fall to the degraded
    # fallback name on import. Pin the contract so a future format drift fails
    # here loudly instead of silently downgrading every imported snapshot.
    from lib.snapshot import snapshot_stem

    match = mod.STEM_RE.match(snapshot_stem("camera-1"))
    assert match is not None
    assert match["camera"] == "camera-1"


def test_collect_folders_includes_bird_and_snapshots_by_default(tmp_path):
    roster = tmp_path / "roster.yaml"
    # Include a roster entry that repeats "bird" to prove de-duplication holds.
    roster.write_text(
        "labels:\n  - { name: matcha }\n  - { name: bird }\n",
        encoding="utf-8",
    )

    folders = mod.collect_folders(roster)

    # Generic "bird" and on-demand "snapshots" lead, roster labels follow.
    assert folders[:2] == ["bird", "snapshots"]
    assert "matcha" in folders
    assert folders.count("bird") == 1
    assert folders.count("snapshots") == 1


# --- suggestion_box (pure) --------------------------------------------------


def test_suggestion_box_normalizes_bbox_and_labels():
    b = mod.suggestion_box(_detection(width=100, height=200, conf=0.7), "matcha")
    assert b["id"] == "sug-0"
    # center (30, 70) / (100, 200); size (40, 100) / (100, 200)
    assert b["cx"] == 0.3 and b["cy"] == 0.35
    assert b["w"] == 0.4 and b["h"] == 0.5
    assert b["conf"] == 0.7
    assert b["label"] == "matcha"
    assert b["labelConf"] == 0.7


def test_suggestion_box_bird_folder_is_unlabeled():
    b = mod.suggestion_box(_detection(), None)
    assert b["label"] is None
    assert b["labelConf"] is None


def test_suggestion_box_defaults_conf_when_missing():
    det = _detection()
    del det["detection"]["confidence"]
    b = mod.suggestion_box(det, "percy")
    assert b["conf"] == 1.0
    assert b["labelConf"] == 1.0


def test_suggestion_box_none_without_bbox():
    assert mod.suggestion_box({"frame": {"width": 1, "height": 1}}, "percy") is None
    assert mod.suggestion_box(_detection(width=0), "percy") is None


# --- classify_and_apply writes a suggestion sidecar, not .txt ----------------


def test_classify_and_apply_writes_suggestion_not_txt(tmp_path):
    src = tmp_path / "src"
    day = tmp_path / "out" / "day"
    ir = tmp_path / "out" / "ir"
    for d in (src, day, ir):
        d.mkdir(parents=True)

    img = src / "20240101_120000_0_camera-1_bird_abc123.jpg"
    _color_jpg(img)
    detection = _detection(conf=0.9)
    img.with_suffix(".json").write_text(json.dumps(detection), encoding="utf-8")

    record = mod.classify_and_apply(img, _args(), day, ir, class_name="matcha")

    assert record["lighting"] == "day"
    assert record["label"] is True  # a suggestion was written

    stem = mod.target_stem(img.stem, "day", detection)
    moved = day / f"{stem}.jpg"
    assert moved.exists()
    # No YOLO .txt seed anymore.
    assert not (day / f"{stem}.txt").exists()
    # The source detection .json was consumed.
    assert not img.with_suffix(".json").exists()

    sug = read_suggestions(moved)
    assert len(sug["boxes"]) == 1
    assert sug["boxes"][0]["label"] == "matcha"
    assert sug["boxes"][0]["conf"] == 0.9
    assert sug["labels"] == []
    assert suggestion_path(moved).exists()


def test_classify_and_apply_bird_folder_leaves_label_none(tmp_path):
    src = tmp_path / "src"
    day = tmp_path / "out" / "day"
    ir = tmp_path / "out" / "ir"
    for d in (src, day, ir):
        d.mkdir(parents=True)

    img = src / "20240101_120000_0_camera-1_bird_abc123.jpg"
    _color_jpg(img)
    detection = _detection()
    img.with_suffix(".json").write_text(json.dumps(detection), encoding="utf-8")

    mod.classify_and_apply(img, _args(), day, ir, class_name="bird")

    moved = day / f"{mod.target_stem(img.stem, 'day', detection)}.jpg"
    sug = read_suggestions(moved)
    assert sug["boxes"][0]["label"] is None


def test_classify_and_apply_snapshot_without_json_moves_without_suggestion(tmp_path):
    src = tmp_path / "src"
    day = tmp_path / "out" / "day"
    ir = tmp_path / "out" / "ir"
    for d in (src, day, ir):
        d.mkdir(parents=True)

    # A /snapshot capture: a color frame with NO paired detection json.
    img = src / "20240101_120000_0_camera-1_snapshot_abc123.jpg"
    _color_jpg(img)

    record = mod.classify_and_apply(img, _args(), day, ir, class_name="snapshots")

    assert record["lighting"] == "day"
    assert record["label"] is False  # no detection => no suggestion sidecar
    assert record["skipped"] is False

    moved = day / f"{mod.target_stem(img.stem, 'day', None)}.jpg"
    assert moved.exists()
    assert not suggestion_path(moved).exists()


def test_classify_and_apply_no_labels_moves_only(tmp_path):
    src = tmp_path / "src"
    day = tmp_path / "out" / "day"
    ir = tmp_path / "out" / "ir"
    for d in (src, day, ir):
        d.mkdir(parents=True)

    img = src / "20240101_120000_0_camera-1_bird_abc123.jpg"
    _color_jpg(img)
    detection = _detection()
    img.with_suffix(".json").write_text(json.dumps(detection), encoding="utf-8")

    record = mod.classify_and_apply(img, _args(no_labels=True), day, ir, class_name="matcha")

    moved = day / f"{mod.target_stem(img.stem, 'day', detection)}.jpg"
    assert moved.exists()
    assert record["label"] is False
    assert not suggestion_path(moved).exists()
