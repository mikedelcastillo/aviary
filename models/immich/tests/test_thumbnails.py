"""Tests for aviary_immich.thumbnails: cache resolution + single-thread chunk download.

All I/O hits tmp_path via FakeImmichClient, which actually writes thumbnail bytes to disk so
the cache/rename logic is exercised for real. decode is kept False throughout so cv2 is never
imported.
"""

from __future__ import annotations

from aviary_immich.thumbnails import (
    cache_thumbnail_chunk,
    delete_cached_thumbnail,
    thumbnail_for_asset,
)
from fakes import FakeImmichClient, make_asset


# --------------------------------------------------------------------------- thumbnail_for_asset


def test_returns_cached_jpg_without_downloading(tmp_path):
    slug = "acct"
    account_cache = tmp_path / slug
    account_cache.mkdir(parents=True)
    cached_file = account_cache / "a1.jpg"
    cached_file.write_bytes(b"already-here")

    client = FakeImmichClient()
    result = thumbnail_for_asset(client, tmp_path, slug, "a1", "preview")

    assert result == cached_file
    assert client.downloaded == []


def test_cache_hit_for_each_known_suffix(tmp_path):
    slug = "acct"
    account_cache = tmp_path / slug
    account_cache.mkdir(parents=True)
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"):
        asset_id = f"asset{suffix}"
        cached_file = account_cache / f"{asset_id}{suffix}"
        cached_file.write_bytes(b"x")
        client = FakeImmichClient()
        result = thumbnail_for_asset(client, tmp_path, slug, asset_id, "preview")
        assert result == cached_file
        assert client.downloaded == []


def test_miss_downloads_then_renames_with_png_suffix(tmp_path):
    slug = "acct"
    client = FakeImmichClient(thumbnail_content_type="image/png")

    result = thumbnail_for_asset(client, tmp_path, slug, "a1", "preview")

    assert result == tmp_path / slug / "a1.png"
    assert result.exists()
    assert client.downloaded == ["a1"]
    # The .download temp file was renamed away, not left behind.
    assert not (tmp_path / slug / "a1.download").exists()


def test_miss_default_jpeg_suffix(tmp_path):
    slug = "acct"
    client = FakeImmichClient(thumbnail_content_type="image/jpeg")

    result = thumbnail_for_asset(client, tmp_path, slug, "a1", "preview")

    assert result == tmp_path / slug / "a1.jpg"
    assert result.exists()
    assert client.downloaded == ["a1"]


def test_miss_creates_account_cache_dir(tmp_path):
    slug = "newslug"
    assert not (tmp_path / slug).exists()
    client = FakeImmichClient()

    result = thumbnail_for_asset(client, tmp_path, slug, "a1", "preview")

    assert (tmp_path / slug).is_dir()
    assert result.exists()


def test_miss_passes_size_through_to_download(tmp_path):
    slug = "acct"
    captured: dict[str, str] = {}

    class RecordingClient(FakeImmichClient):
        def download_thumbnail(self, asset_id, path, size="preview"):
            captured["size"] = size
            return super().download_thumbnail(asset_id, path, size=size)

    client = RecordingClient()
    thumbnail_for_asset(client, tmp_path, slug, "a1", "thumbnail")

    assert captured["size"] == "thumbnail"


def test_returned_file_holds_downloaded_bytes(tmp_path):
    slug = "acct"
    client = FakeImmichClient(thumbnail_bytes=b"PNGDATA", thumbnail_content_type="image/png")

    result = thumbnail_for_asset(client, tmp_path, slug, "a1", "preview")

    assert result.read_bytes() == b"PNGDATA"


# --------------------------------------------------------------------------- cache_thumbnail_chunk


def test_chunk_single_thread_returns_cached_pairs(tmp_path):
    slug = "acct"
    assets = [make_asset(id="a1"), make_asset(id="a2")]
    client = FakeImmichClient(thumbnail_content_type="image/png")

    cached, errors = cache_thumbnail_chunk(
        assets,
        lambda: client,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        decode=False,
    )

    assert errors == []
    assert len(cached) == 2
    returned_assets = [asset for asset, _ in cached]
    assert returned_assets == assets
    for asset, path in cached:
        assert path == tmp_path / slug / f"{asset['id']}.png"
        assert path.exists()


def test_chunk_progress_advance_called_once_per_asset(tmp_path):
    slug = "acct"
    assets = [make_asset(id=f"a{i}") for i in range(3)]
    client = FakeImmichClient()
    calls: list[int] = []

    cache_thumbnail_chunk(
        assets,
        lambda: client,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        progress_advance=lambda n: calls.append(n),
        decode=False,
    )

    assert calls == [1, 1, 1]


def test_chunk_progress_advance_none_is_ok(tmp_path):
    slug = "acct"
    assets = [make_asset(id="a1")]
    client = FakeImmichClient()

    cached, errors = cache_thumbnail_chunk(
        assets,
        lambda: client,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        progress_advance=None,
        decode=False,
    )

    assert len(cached) == 1
    assert errors == []


def test_chunk_per_asset_failure_recorded_not_raised(tmp_path):
    slug = "acct"
    assets = [make_asset(id="boom")]

    class FailingClient(FakeImmichClient):
        def download_thumbnail(self, asset_id, path, size="preview"):
            raise RuntimeError("network down")

    client = FailingClient()

    cached, errors = cache_thumbnail_chunk(
        assets,
        lambda: client,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        decode=False,
    )

    assert cached == []
    assert len(errors) == 1
    record = errors[0]
    assert record["decision"] == "error"
    assert record["asset_id"] == "boom"
    assert record["account"] == slug
    assert record["error"] == "network down"


def test_chunk_mixed_success_and_failure(tmp_path):
    slug = "acct"
    good = make_asset(id="good")
    bad = make_asset(id="bad")

    class SelectiveClient(FakeImmichClient):
        def download_thumbnail(self, asset_id, path, size="preview"):
            if str(asset_id) == "bad":
                raise RuntimeError("kaboom")
            return super().download_thumbnail(asset_id, path, size=size)

    client = SelectiveClient()
    calls: list[int] = []

    cached, errors = cache_thumbnail_chunk(
        [good, bad],
        lambda: client,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        progress_advance=lambda n: calls.append(n),
        decode=False,
    )

    assert len(cached) == 1
    assert cached[0][0] == good
    assert len(errors) == 1
    assert errors[0]["asset_id"] == "bad"
    # progress still advances once per asset, success or failure.
    assert calls == [1, 1]


def test_chunk_factory_called_per_asset(tmp_path):
    slug = "acct"
    assets = [make_asset(id=f"a{i}") for i in range(4)]
    client = FakeImmichClient()
    factory_calls = {"n": 0}

    def factory():
        factory_calls["n"] += 1
        return client

    cache_thumbnail_chunk(
        assets,
        factory,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        decode=False,
    )

    assert factory_calls["n"] == 4


def test_chunk_empty_assets_returns_empty(tmp_path):
    slug = "acct"
    client = FakeImmichClient()

    cached, errors = cache_thumbnail_chunk(
        [],
        lambda: client,
        tmp_path,
        slug,
        "preview",
        download_workers=1,
        decode=False,
    )

    assert cached == []
    assert errors == []


# --------------------------------------------------------------------------- delete_cached_thumbnail


def test_delete_removes_cached_file_for_each_suffix(tmp_path):
    slug = "acct"
    account_cache = tmp_path / slug
    account_cache.mkdir(parents=True)
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"):
        asset_id = f"asset{suffix}"
        cached = account_cache / f"{asset_id}{suffix}"
        cached.write_bytes(b"x")
        delete_cached_thumbnail(tmp_path, slug, asset_id)
        assert not cached.exists()


def test_delete_removes_leftover_download_temp(tmp_path):
    slug = "acct"
    account_cache = tmp_path / slug
    account_cache.mkdir(parents=True)
    temp = account_cache / "a1.download"
    temp.write_bytes(b"partial")

    delete_cached_thumbnail(tmp_path, slug, "a1")

    assert not temp.exists()


def test_delete_missing_file_is_noop(tmp_path):
    # No account dir at all — best-effort cleanup must never raise.
    delete_cached_thumbnail(tmp_path, "acct", "ghost")


def test_delete_only_targets_the_named_asset(tmp_path):
    slug = "acct"
    account_cache = tmp_path / slug
    account_cache.mkdir(parents=True)
    keep = account_cache / "keep.jpg"
    keep.write_bytes(b"keep")
    drop = account_cache / "drop.jpg"
    drop.write_bytes(b"drop")

    delete_cached_thumbnail(tmp_path, slug, "drop")

    assert not drop.exists()
    assert keep.exists()
