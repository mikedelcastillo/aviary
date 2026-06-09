"""Tests for the ``clear-cache`` console script implementation."""

from __future__ import annotations

from aviary_immich.clear_cache import _human, clear_cache


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
