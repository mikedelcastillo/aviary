"""Thumbnail cache resolution and concurrent download/decode for Immich assets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from aviary_immich.records import record_from_error


# Cached thumbnails are named "{asset_id}{suffix}"; probe these known suffixes (jpg first — the
# Immich "preview" default and ~100% of the existing cache) with O(1) exists() instead of a
# per-asset directory glob. The old glob scandir'd the whole account cache (~12-37 ms each on a
# 15k-file dir, GIL-held) on every asset; a few stat() calls are ~1000x cheaper and, unlike the
# glob, do not degrade to O(n^2) as a cold scan fills the directory.
_CACHE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")


def thumbnail_for_asset(client, cache_dir: Path, account_slug: str, asset_id: str, size: str) -> Path:
    from aviary_immich.client import suffix_from_headers

    account_cache = cache_dir / account_slug
    for suffix in _CACHE_SUFFIXES:
        candidate = account_cache / f"{asset_id}{suffix}"
        if candidate.exists():
            return candidate

    account_cache.mkdir(parents=True, exist_ok=True)
    temp_path = account_cache / f"{asset_id}.download"
    headers = client.download_thumbnail(asset_id, temp_path, size=size)
    final_path = account_cache / f"{asset_id}{suffix_from_headers(headers)}"
    temp_path.replace(final_path)
    return final_path


def cache_thumbnail_chunk(
    assets: list[dict[str, Any]],
    client_factory: Callable[[], Any],
    cache_dir: Path,
    account_slug: str,
    thumbnail_size: str,
    download_workers: int = 1,
    progress_advance: Callable[[int], None] | None = None,
    decode: bool = False,
) -> tuple[list[tuple[dict[str, Any], Any]], list[dict[str, Any]]]:
    """Resolve (download if needed) each asset's thumbnail across ``download_workers`` threads.

    With ``decode=True`` each worker also runs ``cv2.imread`` and the returned pairs carry the
    decoded BGR array instead of the path. cv2 releases the GIL, so decoding here — on the
    download pool — overlaps GPU inference instead of serializing on the single GPU thread.
    """
    cached: list[tuple[dict[str, Any], Any]] = []
    errors: list[dict[str, Any]] = []
    if decode:
        import cv2

    def fetch(asset: dict[str, Any]):
        client = client_factory()
        try:
            thumbnail_path = thumbnail_for_asset(client, cache_dir, account_slug, str(asset["id"]), thumbnail_size)
            if decode:
                image = cv2.imread(str(thumbnail_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode thumbnail {thumbnail_path}")
                return asset, image, None
            return asset, thumbnail_path, None
        except Exception as exc:  # noqa: BLE001 - recorded per asset, not fatal
            return asset, None, exc

    def record(asset: dict[str, Any], path: Path | None, exc: Exception | None) -> None:
        if exc is None:
            cached.append((asset, path))
        else:
            errors.append(record_from_error(asset, account_slug, exc))
        if progress_advance is not None:
            progress_advance(1)

    if download_workers <= 1:
        for asset in assets:
            record(*fetch(asset))
    else:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [executor.submit(fetch, asset) for asset in assets]
            for future in as_completed(futures):
                record(*future.result())

    return cached, errors
