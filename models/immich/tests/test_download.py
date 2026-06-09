"""Tests for aviary_immich.download: the pure download_destination helper, parse_args
defaults/flags, and the shell entry-point wiring. main() is intentionally not exercised."""

from __future__ import annotations

from pathlib import Path

import pytest

from aviary_immich.download import download_destination, parse_args
from fakes import make_asset


# --------------------------------------------------------------------------- download_destination

ACCOUNT_DIR = Path("/out")


def test_destination_uses_original_file_name():
    filename, destination = download_destination(make_asset(id="a1", name="photo.jpg"), ACCOUNT_DIR)
    assert filename == "a1_photo.jpg"
    assert destination == ACCOUNT_DIR / "a1_photo.jpg"


def test_destination_prefers_name_over_path():
    asset = make_asset(id="a1", name="photo.jpg", path="/library/ignored.png")
    filename, destination = download_destination(asset, ACCOUNT_DIR)
    assert filename == "a1_photo.jpg"
    assert destination == ACCOUNT_DIR / "a1_photo.jpg"


def test_destination_falls_back_to_path_basename():
    asset = make_asset(id="a1", path="/library/2024/sunset.png")
    filename, destination = download_destination(asset, ACCOUNT_DIR)
    assert filename == "a1_sunset.png"
    assert destination == ACCOUNT_DIR / "a1_sunset.png"


def test_destination_with_neither_name_nor_path():
    # Base name is "{id}_" which is non-empty after sanitizing, so safe_filename returns it
    # (NOT the "{id}.jpg" fallback). The trailing underscore is preserved.
    filename, destination = download_destination(make_asset(id="abc"), ACCOUNT_DIR)
    assert filename == "abc_"
    assert destination == ACCOUNT_DIR / "abc_"


def test_destination_returns_filename_and_path_pair():
    result = download_destination(make_asset(id="a1", name="a.jpg"), ACCOUNT_DIR)
    filename, destination = result
    assert isinstance(filename, str)
    assert isinstance(destination, Path)
    # The destination's filename component matches the returned filename.
    assert destination.name == filename


def test_destination_sanitizes_unsafe_characters_in_name():
    asset = make_asset(id="a1", name="my photo:weird*.jpg")
    filename, destination = download_destination(asset, ACCOUNT_DIR)
    # Colon and asterisk become underscores; spaces/dots/dashes are kept.
    assert filename == "a1_my photo_weird_.jpg"
    assert destination == ACCOUNT_DIR / filename


def test_destination_slashes_in_name_become_underscores():
    asset = make_asset(id="a1", name="sub/dir/file.jpg")
    filename, _ = download_destination(asset, ACCOUNT_DIR)
    assert filename == "a1_sub_dir_file.jpg"
    assert "/" not in filename


def test_header_filename_overrides_name():
    asset = make_asset(id="a1", name="original.jpg")
    filename, destination = download_destination(asset, ACCOUNT_DIR, header_filename="served.png")
    assert filename == "a1_served.png"
    assert destination == ACCOUNT_DIR / "a1_served.png"


def test_header_filename_overrides_even_without_original_name():
    asset = make_asset(id="abc")
    filename, destination = download_destination(asset, ACCOUNT_DIR, header_filename="served.png")
    assert filename == "abc_served.png"
    assert destination == ACCOUNT_DIR / "abc_served.png"


def test_header_filename_is_sanitized():
    asset = make_asset(id="a1", name="orig.jpg")
    filename, destination = download_destination(asset, ACCOUNT_DIR, header_filename="bad/name:x.png")
    assert filename == "a1_bad_name_x.png"
    assert destination == ACCOUNT_DIR / filename


def test_empty_header_filename_does_not_override():
    # header_filename="" is falsy, so the base name from originalFileName is kept.
    asset = make_asset(id="a1", name="keep.jpg")
    filename, destination = download_destination(asset, ACCOUNT_DIR, header_filename="")
    assert filename == "a1_keep.jpg"
    assert destination == ACCOUNT_DIR / "a1_keep.jpg"


def test_destination_coerces_non_string_id():
    # The helper str()-coerces the id; an int id must still produce a usable name.
    asset = make_asset(id=42, name="x.jpg")
    filename, _ = download_destination(asset, ACCOUNT_DIR)
    assert filename == "42_x.jpg"


def test_destination_respects_account_dir():
    other = Path("/some/other/dir")
    filename, destination = download_destination(make_asset(id="a1", name="a.jpg"), other)
    assert destination == other / filename
    assert destination.parent == other


# --------------------------------------------------------------------------- parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds"])
    args = parse_args()
    assert args.page_size == 250
    assert args.output_dir == Path("models/annotation/raw/immich_birds")
    assert args.accounts_config == Path("models/immich/config/accounts.yaml")
    assert args.env_file == Path(".env")
    assert args.manifest_dir == Path("models/immich/data/manifests")
    assert args.limit is None
    assert args.overwrite is False
    assert args.allow_shared_album is False


def test_parse_args_overwrite_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds", "--overwrite"])
    args = parse_args()
    assert args.overwrite is True
    # Other booleans untouched.
    assert args.allow_shared_album is False


def test_parse_args_allow_shared_album_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds", "--allow-shared-album"])
    args = parse_args()
    assert args.allow_shared_album is True
    assert args.overwrite is False


def test_parse_args_both_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds", "--overwrite", "--allow-shared-album"])
    args = parse_args()
    assert args.overwrite is True
    assert args.allow_shared_album is True


def test_parse_args_page_size_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds", "--page-size", "10"])
    args = parse_args()
    assert args.page_size == 10
    assert isinstance(args.page_size, int)


def test_parse_args_output_dir_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds", "--output-dir", "/tmp/birds"])
    args = parse_args()
    assert args.output_dir == Path("/tmp/birds")
    assert isinstance(args.output_dir, Path)


def test_parse_args_limit_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download-birds", "--limit", "5"])
    args = parse_args()
    assert args.limit == 5


# --------------------------------------------------------------------------- shell wiring


def test_shell_main_is_download_main():
    import download_immich_birds
    import aviary_immich.download as d

    assert download_immich_birds.main is d.main
