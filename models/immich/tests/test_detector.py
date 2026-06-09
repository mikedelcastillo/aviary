"""Unit tests for the pure helpers in ``aviary_immich.detector``.

These never construct a real :class:`PretrainedBirdDetector` (which would import ultralytics/YOLO
and load weights). Instead they exercise the module-level functions and a bare ``__new__`` instance,
injecting a fake ``torch`` module where the code consults it.
"""

from __future__ import annotations

import sys
import types

import pytest

from aviary_immich.detector import (
    BirdPrediction,
    PretrainedBirdDetector,
    _should_use_fp16,
    select_device,
)


# --------------------------------------------------------------------------- helpers


def make_fake_torch(
    *,
    hip=None,
    cuda_available=False,
    device_capability=(6, 1),
    mps_available=False,
) -> types.ModuleType:
    """Build a minimal fake ``torch`` module covering the attributes the detector touches."""
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=hip)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_capability=lambda device=None: device_capability,
    )
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available)
    )
    return torch


def box(cls_id, conf, xyxy):
    """Build a fake ultralytics box matching ``cls[0].item()`` / ``conf[0].item()`` / ``xyxy[0].tolist()``."""
    return types.SimpleNamespace(
        cls=[types.SimpleNamespace(item=lambda v=cls_id: v)],
        conf=[types.SimpleNamespace(item=lambda v=conf: v)],
        xyxy=[types.SimpleNamespace(tolist=lambda v=list(xyxy): v)],
    )


def bare_detector() -> PretrainedBirdDetector:
    """A detector instance with no __init__ run (no YOLO load)."""
    return PretrainedBirdDetector.__new__(PretrainedBirdDetector)


# --------------------------------------------------------------------------- _should_use_fp16: env override


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_should_use_fp16_override_falsey_returns_false(monkeypatch, value):
    monkeypatch.setenv("IMMICH_BIRD_HALF", value)
    # Even on a cuda device, the override short-circuits before importing torch.
    assert _should_use_fp16("cuda:0") is False


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_should_use_fp16_override_truthy_returns_true(monkeypatch, value):
    monkeypatch.setenv("IMMICH_BIRD_HALF", value)
    # Even on cpu, the override forces True before any device/torch check.
    assert _should_use_fp16("cpu") is True


def test_should_use_fp16_override_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("IMMICH_BIRD_HALF", "TRUE")
    assert _should_use_fp16("cpu") is True


def test_should_use_fp16_override_is_whitespace_stripped(monkeypatch):
    monkeypatch.setenv("IMMICH_BIRD_HALF", "  off  ")
    assert _should_use_fp16("cuda:0") is False


# --------------------------------------------------------------------------- _should_use_fp16: auto


def test_should_use_fp16_auto_non_cuda_device_returns_false(monkeypatch):
    monkeypatch.delenv("IMMICH_BIRD_HALF", raising=False)
    assert _should_use_fp16("cpu") is False
    assert _should_use_fp16("mps") is False


def test_should_use_fp16_explicit_auto_non_cuda_returns_false(monkeypatch):
    monkeypatch.setenv("IMMICH_BIRD_HALF", "auto")
    assert _should_use_fp16("cpu") is False


def test_should_use_fp16_auto_cuda_torch_import_fails_returns_false(monkeypatch):
    monkeypatch.delenv("IMMICH_BIRD_HALF", raising=False)
    # Setting the module to None makes `import torch` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _should_use_fp16("cuda:0") is False


def test_should_use_fp16_auto_cuda_rocm_hip_returns_true(monkeypatch):
    monkeypatch.delenv("IMMICH_BIRD_HALF", raising=False)
    fake = make_fake_torch(hip="6.0.0")
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert _should_use_fp16("cuda:0") is True


def test_should_use_fp16_auto_cuda_capability_7_returns_true(monkeypatch):
    monkeypatch.delenv("IMMICH_BIRD_HALF", raising=False)
    fake = make_fake_torch(hip=None, device_capability=(7, 0))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert _should_use_fp16("cuda:0") is True


def test_should_use_fp16_auto_cuda_capability_6_returns_false(monkeypatch):
    monkeypatch.delenv("IMMICH_BIRD_HALF", raising=False)
    fake = make_fake_torch(hip=None, device_capability=(6, 1))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert _should_use_fp16("cuda:0") is False


# --------------------------------------------------------------------------- select_device


def test_select_device_explicit_preferred_returned_verbatim():
    assert select_device("cuda:1") == "cuda:1"
    assert select_device("cpu") == "cpu"
    assert select_device("mps") == "mps"


def test_select_device_auto_torch_import_fails_returns_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_device("auto") == "cpu"


def test_select_device_default_arg_is_auto(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_device() == "cpu"


def test_select_device_auto_cuda_available_returns_cuda0(monkeypatch):
    fake = make_fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert select_device("auto") == "cuda:0"


def test_select_device_auto_mps_available_returns_mps(monkeypatch):
    fake = make_fake_torch(cuda_available=False, mps_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert select_device("auto") == "mps"


def test_select_device_auto_neither_returns_cpu(monkeypatch):
    fake = make_fake_torch(cuda_available=False, mps_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert select_device("auto") == "cpu"


def test_select_device_auto_no_mps_backend_returns_cpu(monkeypatch):
    # torch.backends without an mps attribute -> getattr returns None -> cpu.
    fake = make_fake_torch(cuda_available=False)
    fake.backends = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert select_device("auto") == "cpu"


# --------------------------------------------------------------------------- _resolve_bird_class_ids


def test_resolve_bird_class_ids_dict_names():
    det = bare_detector()
    det._names = {0: "person", 14: "bird", 16: "dog"}
    det.bird_labels = {"bird", "dog"}
    assert det._resolve_bird_class_ids() == {14, 16}


def test_resolve_bird_class_ids_list_names():
    det = bare_detector()
    det._names = ["person", "bird", "dog"]
    det.bird_labels = {"bird"}
    assert det._resolve_bird_class_ids() == {1}


def test_resolve_bird_class_ids_is_case_insensitive():
    det = bare_detector()
    det._names = {0: "Person", 14: "BIRD", 16: "Dog"}
    det.bird_labels = {"bird", "dog"}
    assert det._resolve_bird_class_ids() == {14, 16}


def test_resolve_bird_class_ids_no_match_returns_empty():
    det = bare_detector()
    det._names = {0: "person", 16: "dog"}
    det.bird_labels = {"bird"}
    assert det._resolve_bird_class_ids() == set()


# --------------------------------------------------------------------------- _prediction_from_result


def test_prediction_from_result_single_bird_detection():
    det = bare_detector()
    det._names = {14: "bird"}
    result = types.SimpleNamespace(boxes=[box(14, 0.91, [1.4, 2.6, 9.9, 8.1])])

    prediction = det._prediction_from_result(result)

    assert isinstance(prediction, BirdPrediction)
    assert prediction.has_bird is True
    assert prediction.max_confidence == 0.91
    assert len(prediction.detections) == 1

    detection = prediction.detections[0]
    assert detection["label"] == "bird"
    assert detection["confidence"] == 0.91
    # xyxy rounded to nearest int: 1.4->1, 2.6->3, 9.9->10, 8.1->8
    assert (detection["x1"], detection["y1"], detection["x2"], detection["y2"]) == (1, 3, 10, 8)


def test_prediction_from_result_empty_boxes():
    det = bare_detector()
    det._names = {14: "bird"}
    result = types.SimpleNamespace(boxes=[])

    prediction = det._prediction_from_result(result)

    assert prediction.has_bird is False
    assert prediction.max_confidence == 0.0
    assert prediction.detections == []


def test_prediction_from_result_max_confidence_across_boxes():
    det = bare_detector()
    det._names = {14: "bird", 16: "dog"}
    result = types.SimpleNamespace(
        boxes=[
            box(14, 0.30, [0, 0, 10, 10]),
            box(16, 0.85, [5, 5, 20, 20]),
            box(14, 0.50, [1, 1, 2, 2]),
        ]
    )

    prediction = det._prediction_from_result(result)

    assert prediction.has_bird is True
    assert prediction.max_confidence == 0.85
    assert len(prediction.detections) == 3
    assert [d["label"] for d in prediction.detections] == ["bird", "dog", "bird"]


def test_prediction_from_result_list_names_label_lookup():
    det = bare_detector()
    det._names = ["person", "bird", "dog"]
    result = types.SimpleNamespace(boxes=[box(2, 0.7, [0, 0, 1, 1])])

    prediction = det._prediction_from_result(result)

    assert prediction.detections[0]["label"] == "dog"


def test_prediction_from_result_dict_missing_class_id_falls_back_to_id():
    # When names is a dict and the class id is absent, names.get returns the id itself,
    # which is then str()'d into the label.
    det = bare_detector()
    det._names = {14: "bird"}
    result = types.SimpleNamespace(boxes=[box(99, 0.42, [0, 0, 4, 4])])

    prediction = det._prediction_from_result(result)

    assert prediction.detections[0]["label"] == "99"
