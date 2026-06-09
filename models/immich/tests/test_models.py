"""Tests for ``aviary_immich.models``: the ``build_models`` factory and YOLO output mapping.

CPU-light: we never actually instantiate ``YoloObjectModel`` (that would load
ultralytics/torch). ``build_models`` is exercised via a monkeypatched fake, and ``_to_output`` is
called as a staticmethod with no detector behind it.
"""

from __future__ import annotations

import fakes
import pytest

from aviary_immich.config import ModelSpec
from aviary_immich.models import build_models
from aviary_immich.models.base import ModelOutput, Tag
from aviary_immich.models.yolo import YoloObjectModel


# --------------------------------------------------------------------------- build_models


class FakeYolo:
    """Captures its constructor args instead of loading the real detector."""

    def __init__(self, model_name, threshold, device, labels):
        self.init_args = (model_name, threshold, device, labels)
        self.name = "yolo"
        self.half = False


def test_build_models_constructs_yolo_with_default_threshold(monkeypatch):
    monkeypatch.setattr("aviary_immich.models.yolo.YoloObjectModel", FakeYolo)
    spec = ModelSpec(name="yolo", kind="yolo", labels=("bird",))
    models = build_models((spec,), "cpu", "weights.pt", 0.30)
    assert len(models) == 1
    assert models[0].init_args == ("weights.pt", 0.30, "cpu", ("bird",))


def test_build_models_spec_threshold_overrides_default(monkeypatch):
    monkeypatch.setattr("aviary_immich.models.yolo.YoloObjectModel", FakeYolo)
    spec = ModelSpec(name="yolo", kind="yolo", labels=("bird",), threshold=0.7)
    models = build_models((spec,), "cpu", "weights.pt", 0.30)
    assert models[0].init_args == ("weights.pt", 0.7, "cpu", ("bird",))


def test_build_models_skips_disabled_specs(monkeypatch):
    monkeypatch.setattr("aviary_immich.models.yolo.YoloObjectModel", FakeYolo)
    enabled = ModelSpec(name="yolo", kind="yolo", labels=("bird",))
    disabled = ModelSpec(name="yolo2", kind="yolo", labels=("dog",), enabled=False)
    models = build_models((enabled, disabled), "cpu", "weights.pt", 0.30)
    assert len(models) == 1
    assert models[0].init_args == ("weights.pt", 0.30, "cpu", ("bird",))


def test_build_models_unknown_kind_raises():
    spec = ModelSpec(name="mystery", kind="bogus")
    with pytest.raises(ValueError):
        build_models((spec,), "cpu", "weights.pt", 0.30)


class FakeClip:
    """Captures CLIP constructor kwargs instead of loading open_clip/torch."""

    def __init__(self, prompts, threshold, device, **options):
        self.init = {"prompts": prompts, "threshold": threshold, "device": device, "options": options}
        self.name = "clip"
        self.half = False


def test_build_models_constructs_clip_from_spec(monkeypatch):
    # CLIP is built only when its spec is enabled; build_models lazy-imports the backend, so
    # patching the class on its module is enough (no open_clip/torch needed).
    monkeypatch.setattr("aviary_immich.models.clip.ClipSceneModel", FakeClip)
    spec = ModelSpec(
        name="clip",
        kind="clip",
        prompts={"tennis court": ("a tennis court",)},
        threshold=0.28,
        options={"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    )
    models = build_models((spec,), "cpu", "weights.pt", 0.30)
    assert len(models) == 1
    assert models[0].init == {
        "prompts": {"tennis court": ("a tennis court",)},
        "threshold": 0.28,
        "device": "cpu",
        "options": {"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    }


def test_build_models_mixed_specs_preserve_order(monkeypatch):
    monkeypatch.setattr("aviary_immich.models.yolo.YoloObjectModel", FakeYolo)
    monkeypatch.setattr("aviary_immich.models.clip.ClipSceneModel", FakeClip)
    yolo = ModelSpec(name="yolo", kind="yolo", labels=("bird",))
    clip = ModelSpec(name="clip", kind="clip", prompts={"tennis court": ("a tennis court",)}, threshold=0.28)
    models = build_models((yolo, clip), "cpu", "weights.pt", 0.30)
    assert [m.name for m in models] == ["yolo", "clip"]


# --------------------------------------------------------------------------- _to_output


def test_to_output_maps_prediction_to_model_output():
    pred = fakes.make_prediction(labels=["bird"], confidence=0.8)
    output = YoloObjectModel._to_output(pred)
    assert isinstance(output, ModelOutput)
    assert output.tags == (Tag("bird", 0.8, (0, 0, 10, 10)),)
    assert output.max_confidence == 0.8


def test_to_output_lowercases_label_and_carries_box():
    detection = fakes.make_detection("Bird", confidence=0.55, box=(1, 2, 3, 4))
    pred = fakes.make_prediction(detections=[detection])
    output = YoloObjectModel._to_output(pred)
    assert output.tags == (Tag("bird", 0.55, (1, 2, 3, 4)),)
    assert output.max_confidence == 0.55


def test_to_output_empty_prediction():
    pred = fakes.make_prediction(detections=[])
    output = YoloObjectModel._to_output(pred)
    assert output.tags == ()
    assert output.max_confidence == 0.0
