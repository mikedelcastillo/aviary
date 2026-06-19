from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from lib.config import ModelConfig, build_config
from lib.detector import Detection, ObjectDetector


# --- config parsing -------------------------------------------------------


def _base_env(monkeypatch) -> None:
    monkeypatch.setenv("TAPO_CREDENTIALS", "user:password")


def test_multiple_model_paths_parsed(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", "/a.pt, /b.pt")

    config = build_config()

    assert config.model.paths == (Path("/a.pt"), Path("/b.pt"))


def test_single_model_path_is_one_element_tuple(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", "/a.pt")

    assert build_config().model.paths == (Path("/a.pt"),)


def test_missing_model_path_raises(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(ValueError):
        build_config()


def test_blank_model_path_raises(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", " , ")

    with pytest.raises(ValueError):
        build_config()


# --- predict merges passes from every model -------------------------------


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: tuple[float, float, float, float]) -> None:
        self.cls = [types.SimpleNamespace(item=lambda v=cls_id: v)]
        self.conf = [types.SimpleNamespace(item=lambda v=conf: v)]
        self.xyxy = [types.SimpleNamespace(tolist=lambda v=list(xyxy): v)]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeModel:
    def __init__(self, names: dict[int, str], boxes: list[_FakeBox]) -> None:
        self.names = names
        self._boxes = boxes

    def predict(self, **_kwargs):
        return [_FakeResult(self._boxes)]


def _install_fake_yolo(monkeypatch, models: list[_FakeModel]) -> None:
    """Make ``from ultralytics import YOLO`` hand back our fake models in order."""
    queue = list(models)
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = lambda _path: queue.pop(0)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    # Paths don't exist on disk; skip the existence guard.
    monkeypatch.setattr(Path, "exists", lambda self: True)


def test_predict_concatenates_detections_from_all_models(monkeypatch) -> None:
    model_a = _FakeModel({0: "bird"}, [_FakeBox(0, 0.9, (0, 0, 10, 10))])
    model_b = _FakeModel(
        {0: "cat", 1: "dog"},
        [_FakeBox(0, 0.8, (5, 5, 15, 15)), _FakeBox(1, 0.7, (20, 20, 30, 30))],
    )
    _install_fake_yolo(monkeypatch, [model_a, model_b])

    detector = ObjectDetector(ModelConfig(paths=(Path("/a.pt"), Path("/b.pt"))))
    detections = detector.predict(frame=object())

    assert detections == [
        Detection(label="bird", confidence=0.9, bbox_xyxy=(0, 0, 10, 10)),
        Detection(label="cat", confidence=0.8, bbox_xyxy=(5, 5, 15, 15)),
        Detection(label="dog", confidence=0.7, bbox_xyxy=(20, 20, 30, 30)),
    ]


def test_missing_model_file_raises(monkeypatch) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(FileNotFoundError):
        ObjectDetector(ModelConfig(paths=(Path("/missing.pt"),)))
