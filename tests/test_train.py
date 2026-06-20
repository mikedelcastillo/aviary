from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "scripts" / "train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_omit_optional_overrides() -> None:
    mod = _load_module()
    ta = mod.build_train_args(mod.parse_args(["--data", "d.yaml"]))
    assert ta["data"] == str(Path("d.yaml"))
    assert ta["patience"] == 100
    assert ta["exist_ok"] is True
    for absent in ("cls_pw", "hsv_h", "time", "device"):
        assert absent not in ta


def test_cls_pw_passed_through() -> None:
    mod = _load_module()
    ta = mod.build_train_args(mod.parse_args(["--data", "d.yaml", "--cls-pw", "0.5"]))
    assert ta["cls_pw"] == 0.5


def test_hsv_h_zero_is_passed_not_dropped() -> None:
    # 0.0 is falsy but must still be forwarded (it disables hue jitter on purpose).
    mod = _load_module()
    ta = mod.build_train_args(mod.parse_args(["--data", "d.yaml", "--hsv-h", "0"]))
    assert ta["hsv_h"] == 0.0


def test_time_and_patience_overrides() -> None:
    mod = _load_module()
    ta = mod.build_train_args(mod.parse_args(["--data", "d.yaml", "--time", "1.5", "--patience", "30"]))
    assert ta["time"] == 1.5
    assert ta["patience"] == 30


def test_explicit_device_included() -> None:
    mod = _load_module()
    ta = mod.build_train_args(mod.parse_args(["--data", "d.yaml", "--device", "cuda:0"]))
    assert ta["device"] == "cuda:0"
