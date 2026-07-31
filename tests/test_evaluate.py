from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "scripts" / "evaluate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Box:
    map50 = 0.91234
    map = 0.7345
    mp = 0.88
    mr = 0.79


class _Metrics:
    box = _Box()
    speed = {"preprocess": 1.234, "inference": 20.567, "postprocess": 2.0}


def test_summarize_val_extracts_and_rounds() -> None:
    summary = _load_module().summarize_val(_Metrics())
    assert summary["map50"] == 0.9123
    assert summary["map50_95"] == 0.7345
    assert summary["precision"] == 0.88
    assert summary["recall"] == 0.79
    assert summary["speed_ms"] == {"preprocess": 1.23, "inference": 20.57, "postprocess": 2.0}


def test_summarize_val_tolerates_missing_attributes() -> None:
    summary = _load_module().summarize_val(object())
    assert summary == {
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "speed_ms": None,
    }
