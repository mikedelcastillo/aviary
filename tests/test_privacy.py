"""PersonScreen: the Telegram privacy screen must be recall-biased and
fail-closed — anything it can't positively clear counts as containing a person."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.config import PrivacyConfig, _privacy_config
from lib.privacy import PersonScreen


class _FakeBox:
    def __init__(self, cls_id: int, conf: float) -> None:
        self.cls = [types.SimpleNamespace(item=lambda v=cls_id: v)]
        self.conf = [types.SimpleNamespace(item=lambda v=conf: v)]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeModel:
    def __init__(self, names: dict[int, str], boxes: list[_FakeBox]) -> None:
        self.names = names
        self._boxes = boxes
        self.predict_kwargs: dict | None = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [_FakeResult(self._boxes)]


class _RaisingModel:
    names: dict[int, str] = {}

    def predict(self, **_kwargs):
        raise RuntimeError("inference exploded")


def _install_fake_yolo(monkeypatch, model) -> None:
    """Make ``from ultralytics import YOLO`` hand back our fake model."""
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = lambda _path: model
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    # The model path doesn't exist on disk; skip the existence guard.
    monkeypatch.setattr(Path, "exists", lambda self: True)


def _jpeg_bytes() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    return buffer.tobytes()


def _screen(monkeypatch, model) -> PersonScreen:
    _install_fake_yolo(monkeypatch, model)
    return PersonScreen(PrivacyConfig())


# --- person detection ------------------------------------------------------


def test_person_box_flags_image(monkeypatch) -> None:
    screen = _screen(monkeypatch, _FakeModel({0: "person"}, [_FakeBox(0, 0.9)]))
    assert screen.has_person(_jpeg_bytes()) is True


def test_bird_only_image_passes(monkeypatch) -> None:
    screen = _screen(monkeypatch, _FakeModel({0: "bird"}, [_FakeBox(0, 0.9)]))
    assert screen.has_person(_jpeg_bytes()) is False


def test_no_detections_passes(monkeypatch) -> None:
    screen = _screen(monkeypatch, _FakeModel({0: "person"}, []))
    assert screen.has_person(_jpeg_bytes()) is False


def test_person_matched_by_name_not_class_index(monkeypatch) -> None:
    # A non-COCO ordering must still work: person is class 7 here, not 0.
    model = _FakeModel({0: "bird", 7: "person"}, [_FakeBox(7, 0.5)])
    screen = _screen(monkeypatch, model)
    assert screen.has_person(_jpeg_bytes()) is True


def test_predict_uses_configured_thresholds(monkeypatch) -> None:
    model = _FakeModel({0: "person"}, [])
    screen = _screen(monkeypatch, model)
    screen.has_person(_jpeg_bytes())
    assert model.predict_kwargs is not None
    assert model.predict_kwargs["conf"] == PrivacyConfig.confidence
    assert model.predict_kwargs["imgsz"] == PrivacyConfig.image_size
    assert model.predict_kwargs["device"] == PrivacyConfig.device


# --- fail-closed behavior ---------------------------------------------------


def test_undecodable_image_fails_closed(monkeypatch) -> None:
    screen = _screen(monkeypatch, _FakeModel({0: "person"}, []))
    assert screen.has_person(b"not a jpeg") is True


def test_inference_error_fails_closed(monkeypatch) -> None:
    screen = _screen(monkeypatch, _RaisingModel())
    assert screen.has_person(_jpeg_bytes()) is True


def test_missing_model_file_raises_at_construction(tmp_path) -> None:
    # A missing EXPLICIT path fails fast at boot, not fail-closed on every send.
    config = PrivacyConfig(model_path=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        PersonScreen(config)


def test_bare_stock_model_name_defers_to_ultralytics_download(monkeypatch, tmp_path) -> None:
    # *.pt is gitignored, so on a fresh checkout the default "yolo11n.pt" is
    # absent — a bare stock name must reach YOLO (which auto-downloads official
    # weights) instead of failing the boot.
    model = _FakeModel({0: "person"}, [])
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = lambda _path: model
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.chdir(tmp_path)  # guarantee the weights file is absent

    screen = PersonScreen(PrivacyConfig(model_path=Path("yolo11n.pt")))
    assert screen.has_person(_jpeg_bytes()) is False


# --- env config -------------------------------------------------------------


def test_privacy_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PRIVACY_FILTER", raising=False)
    assert _privacy_config().enabled is PrivacyConfig.enabled is True


def test_privacy_filter_env_disables(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_FILTER", "0")
    assert _privacy_config().enabled is False


def test_privacy_model_path_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_MODEL_PATH", "/models/custom.pt")
    assert _privacy_config().model_path == Path("/models/custom.pt")


def test_privacy_model_path_blank_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_MODEL_PATH", "  ")
    assert _privacy_config().model_path == PrivacyConfig.model_path


def test_privacy_confidence_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_CONFIDENCE", "0.4")
    assert _privacy_config().confidence == 0.4


def test_privacy_confidence_invalid_raises(monkeypatch) -> None:
    monkeypatch.setenv("PRIVACY_CONFIDENCE", "very sure")
    with pytest.raises(ValueError):
        _privacy_config()
