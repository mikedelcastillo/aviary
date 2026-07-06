"""Tests for MODEL_CONFIDENCE parsing in lib.config._model_confidence.

Imports only lib.config (no ultralytics), so these stay fast and isolated.
"""

from __future__ import annotations

import os

import pytest

from lib.config import (
    DiscoveryConfig,
    ModelConfig,
    OllamaConfig,
    _model_confidence,
    _model_image_size,
    _ollama_config,
)


def test_confidence_defaults_to_model_config_when_unset(monkeypatch) -> None:
    # No env var -> fall back to the single source of truth, the dataclass default.
    monkeypatch.delenv("MODEL_CONFIDENCE", raising=False)
    assert _model_confidence() == ModelConfig.confidence


def test_default_model_confidence_is_half() -> None:
    # The Python default sits at the F1 peak measured on the held-out test split.
    assert ModelConfig.confidence == 0.5


def test_discovery_workers_capped_for_wide_sweep() -> None:
    # The /discover sweep concurrency is capped at 8 hosts so a wide scan can't
    # crowd the LAN or peg the box (see "cap /discover sweep concurrency at 8").
    assert DiscoveryConfig.max_workers == 8


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


def test_ollama_keep_alive_env_overrides_default(monkeypatch) -> None:
    # OLLAMA_KEEP_ALIVE controls how long workers keep a model resident; the
    # env must reach OllamaConfig.keep_alive so "-1" (forever) etc. take effect.
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "1h")
    assert _ollama_config().keep_alive == "1h"


def test_ollama_keep_alive_blank_env_falls_back_to_default(monkeypatch) -> None:
    # A blank value in .env behaves like "unset" -> the dataclass "30m" default.
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "  ")
    assert OllamaConfig.keep_alive == "30m"
    assert _ollama_config().keep_alive == "30m"


def test_ollama_timeout_seconds_env_overrides_default(monkeypatch) -> None:
    # OLLAMA_TIMEOUT_SECONDS bounds each generation so a runaway can't peg the
    # CPU; the env override must land in OllamaConfig.timeout_seconds.
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45")
    assert _ollama_config().timeout_seconds == 45.0


def test_ollama_timeout_seconds_defaults_when_unset(monkeypatch) -> None:
    # No env var -> fall back to the single source of truth, the 120s default.
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)
    assert OllamaConfig.timeout_seconds == 120.0
    assert _ollama_config().timeout_seconds == 120.0
