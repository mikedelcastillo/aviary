from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "scripts" / "prepare_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves field types via sys.modules[__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _roster(tmp_path: Path) -> Path:
    path = tmp_path / "roster.yaml"
    path.write_text("labels:\n  - { name: percy, models: [live] }\n", encoding="utf-8")
    return path


def _reviewed_image(src: Path, stem: str) -> None:
    (src / f"{stem}.jpg").write_bytes(b"")
    (src / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (src / f"{stem}.json").write_text(
        json.dumps({"boxed": True, "boxes": [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "label": "percy"}]}),
        encoding="utf-8",
    )


def _unreviewed_placeholder(src: Path, stem: str) -> None:
    # Auto-collected: a hardcoded class-0 placeholder .txt, no .json.
    (src / f"{stem}.jpg").write_bytes(b"")
    (src / f"{stem}.txt").write_text("0 0.4 0.7 0.1 0.4\n", encoding="utf-8")


def _train_images(output: Path) -> list[str]:
    return sorted(p.stem for p in (output / "images" / "train").glob("*.jpg"))


def test_main_excludes_unreviewed_by_default(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    _reviewed_image(src, "reviewed_percy")
    _unreviewed_placeholder(src, "auto_draft")
    output = tmp_path / "ds"

    _load_module().main(
        [
            "--source", str(src),
            "--output", str(output),
            "--model", "live",
            "--roster", str(_roster(tmp_path)),
            "--val-ratio", "0",
            "--test-ratio", "0",
        ]
    )

    # Only the human-reviewed frame reaches training; the placeholder is dropped.
    assert _train_images(output) == ["reviewed_percy"]


def test_main_include_unreviewed_keeps_placeholders(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    _reviewed_image(src, "reviewed_percy")
    _unreviewed_placeholder(src, "auto_draft")
    output = tmp_path / "ds"

    _load_module().main(
        [
            "--source", str(src),
            "--output", str(output),
            "--model", "live",
            "--roster", str(_roster(tmp_path)),
            "--val-ratio", "0",
            "--test-ratio", "0",
            "--include-unreviewed",
        ]
    )

    assert _train_images(output) == ["auto_draft", "reviewed_percy"]
