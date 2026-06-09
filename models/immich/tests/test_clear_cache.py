"""Tests for the ``clear-cache`` console script implementation."""

from __future__ import annotations

from aviary_immich.clear_cache import _human, clear_cache, clear_data


def _populate(cache_dir):
    (cache_dir / ".gitkeep").write_text("", encoding="utf-8")
    acct = cache_dir / "account_1"
    acct.mkdir()
    (acct / "a.jpg").write_bytes(b"x" * 100)
    (acct / "b.jpg").write_bytes(b"y" * 50)
    (cache_dir / "loose.jpg").write_bytes(b"z" * 25)


def test_clear_cache_deletes_files_and_reports_totals(tmp_path):
    cache = tmp_path / "thumbnails"
    cache.mkdir()
    _populate(cache)

    files, freed = clear_cache(cache, dry_run=False)

    assert files == 3
    assert freed == 175
    # .gitkeep is preserved; everything else is gone.
    assert (cache / ".gitkeep").exists()
    assert not (cache / "account_1").exists()
    assert not (cache / "loose.jpg").exists()


def test_clear_cache_dry_run_reports_without_deleting(tmp_path):
    cache = tmp_path / "thumbnails"
    cache.mkdir()
    _populate(cache)

    files, freed = clear_cache(cache, dry_run=True)

    assert files == 3
    assert freed == 175
    # Nothing deleted.
    assert (cache / "account_1" / "a.jpg").exists()
    assert (cache / "loose.jpg").exists()


def test_clear_cache_missing_dir_is_noop(tmp_path):
    assert clear_cache(tmp_path / "does-not-exist") == (0, 0)


def test_clear_cache_only_gitkeep_is_empty(tmp_path):
    cache = tmp_path / "thumbnails"
    cache.mkdir()
    (cache / ".gitkeep").write_text("", encoding="utf-8")
    assert clear_cache(cache) == (0, 0)
    assert (cache / ".gitkeep").exists()


def test_human_readable_sizes():
    assert _human(0) == "0.0 B"
    assert _human(512) == "512.0 B"
    assert _human(1536) == "1.5 KB"
    assert _human(5 * 1024**3) == "5.0 GB"


# --------------------------------------------------------------------------- clear_data (full reset)


def _populate_data(data_dir):
    for sub in ("cache", "manifests", "state"):
        sub_dir = data_dir / sub
        sub_dir.mkdir(parents=True)
        (sub_dir / ".gitkeep").write_text("", encoding="utf-8")
    acct = data_dir / "cache" / "acct"
    acct.mkdir()
    (acct / "a.jpg").write_bytes(b"x" * 100)
    (data_dir / "manifests" / "acct_birds.csv").write_bytes(b"y" * 40)
    (data_dir / "state" / "acct_scan.jsonl").write_bytes(b"z" * 60)


def test_clear_data_clears_all_subdirs_preserving_gitkeep(tmp_path):
    data = tmp_path / "data"
    _populate_data(data)

    results = clear_data(data, dry_run=False)

    assert results["cache"] == (1, 100)
    assert results["manifests"] == (1, 40)
    assert results["state"] == (1, 60)
    for sub in ("cache", "manifests", "state"):
        assert (data / sub / ".gitkeep").exists()
    assert not (data / "cache" / "acct").exists()
    assert not (data / "manifests" / "acct_birds.csv").exists()
    assert not (data / "state" / "acct_scan.jsonl").exists()


def test_clear_data_dry_run_reports_without_deleting(tmp_path):
    data = tmp_path / "data"
    _populate_data(data)

    results = clear_data(data, dry_run=True)

    assert results["cache"] == (1, 100)
    assert (data / "cache" / "acct" / "a.jpg").exists()
    assert (data / "state" / "acct_scan.jsonl").exists()


def test_clear_data_skips_missing_subdirs(tmp_path):
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    (data / "state" / "x.jsonl").write_bytes(b"q" * 10)

    results = clear_data(data)

    assert set(results) == {"state"}
    assert results["state"] == (1, 10)
