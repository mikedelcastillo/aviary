from __future__ import annotations

from aviary_training.models import next_version_path


def test_first_version_when_dir_empty(tmp_path) -> None:
    assert next_version_path(tmp_path, "live") == tmp_path / "live-001.pt"


def test_first_version_when_dir_missing(tmp_path) -> None:
    missing = tmp_path / "models"
    assert next_version_path(missing, "live") == missing / "live-001.pt"


def test_increments_past_highest_existing(tmp_path) -> None:
    (tmp_path / "live-001.pt").touch()
    (tmp_path / "live-002.pt").touch()
    assert next_version_path(tmp_path, "live") == tmp_path / "live-003.pt"


def test_uses_max_not_count(tmp_path) -> None:
    # A gap (003 deleted) must not cause a collision with 002.
    (tmp_path / "live-001.pt").touch()
    (tmp_path / "live-004.pt").touch()
    assert next_version_path(tmp_path, "live") == tmp_path / "live-005.pt"


def test_tracks_each_name_independently(tmp_path) -> None:
    (tmp_path / "live-001.pt").touch()
    (tmp_path / "live-002.pt").touch()
    assert next_version_path(tmp_path, "archive") == tmp_path / "archive-001.pt"


def test_ignores_unrelated_and_prefix_collisions(tmp_path) -> None:
    (tmp_path / "live-001.pt").touch()
    (tmp_path / "livestock-009.pt").touch()  # different name, must not match
    (tmp_path / "live-best.pt").touch()  # non-numeric, must not match
    (tmp_path / "live-002.onnx").touch()  # wrong suffix, must not match
    assert next_version_path(tmp_path, "live") == tmp_path / "live-002.pt"


def test_grows_past_three_digits(tmp_path) -> None:
    (tmp_path / "live-999.pt").touch()
    assert next_version_path(tmp_path, "live") == tmp_path / "live-1000.pt"
