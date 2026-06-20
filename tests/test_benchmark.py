from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "scripts" / "benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


def _write_manifest(dataset_dir: Path, rows: list[dict]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "source_image"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_split_sources_returns_only_the_requested_split(tmp_path: Path) -> None:
    bench = _load_module()
    datasets = tmp_path / "datasets"
    img_train = tmp_path / "a.jpg"
    img_test = tmp_path / "b.jpg"
    _write_manifest(
        datasets / "live",
        [
            {"split": "train", "source_image": str(img_train)},
            {"split": "test", "source_image": str(img_test)},
        ],
    )

    allowed = bench.load_split_sources(datasets, "live", "test")

    assert allowed == {img_test.resolve()}


def test_load_split_sources_returns_none_when_manifest_missing(tmp_path: Path) -> None:
    bench = _load_module()
    assert bench.load_split_sources(tmp_path / "datasets", "archive", "test") is None


def test_default_conf_matches_server_model_config() -> None:
    # The benchmark must score at the same confidence the live server uses, so the
    # default tracks server/lib/config.py rather than a drifting hardcoded copy.
    bench = _load_module()
    from lib.config import ModelConfig

    assert bench.parse_args([]).conf == ModelConfig.confidence
