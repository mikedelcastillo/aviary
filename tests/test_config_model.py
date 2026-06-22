"""Tests for MODEL_CONFIDENCE parsing in lib.config._model_confidence.

Imports only lib.config (no ultralytics), so these stay fast and isolated.
"""

from __future__ import annotations

import pytest

from lib.config import ModelConfig, _model_confidence, _model_image_size


def test_confidence_defaults_to_model_config_when_unset(monkeypatch) -> None:
    # No env var -> fall back to the single source of truth, the dataclass default.
    monkeypatch.delenv("MODEL_CONFIDENCE", raising=False)
    assert _model_confidence() == ModelConfig.confidence


def test_default_model_confidence_is_half() -> None:
    # The Python default sits at the F1 peak measured on the held-out test split.
    assert ModelConfig.confidence == 0.5


def test_confidence_env_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONFIDENCE", "0.3")
    assert _model_confidence() == 0.3


def test_confidence_blank_env_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONFIDENCE", "  ")
    assert _model_confidence() == ModelConfig.confidence


def test_confidence_invalid_env_raises(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONFIDENCE", "high")
    with pytest.raises(ValueError):
        _model_confidence()


def test_image_size_defaults_to_model_config_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_IMAGE_SIZE", raising=False)
    assert _model_image_size() == ModelConfig.image_size


def test_image_size_env_overrides_default(monkeypatch) -> None:
    # 1280 is the inference size at which live-007 measured best on the test split.
    monkeypatch.setenv("MODEL_IMAGE_SIZE", "1280")
    assert _model_image_size() == 1280


def test_image_size_blank_env_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_IMAGE_SIZE", "  ")
    assert _model_image_size() == ModelConfig.image_size


def test_image_size_invalid_env_raises(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_IMAGE_SIZE", "huge")
    with pytest.raises(ValueError):
        _model_image_size()
