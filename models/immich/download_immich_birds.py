#!/usr/bin/env python3
"""Download all images from each configured account's Immich Birds album."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-config", type=Path, default=Path("models/immich/config/accounts.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/annotation/raw/immich_birds"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("models/immich/manifests"))
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-shared-album", action="store_true")
    return parser.parse_args()


def progress(items: Iterable[Any], **kwargs) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, **kwargs)


def main() -> None:
    args = parse_args()

    from aviary_immich.client import ImmichClient, filename_from_headers, safe_filename
    from aviary_immich.config import BIRD_ALBUM_NAME, load_accounts_config
    from aviary_immich.state import append_csv, load_manifest_ids, utc_now

    config = load_accounts_config(args.accounts_config, args.env_file)
    manifest_fields = [
        "account",
        "album_id",
        "asset_id",
        "original_file_name",
        "local_path",
        "downloaded_at",
    ]

    for account in config.accounts:
        print(f"\nDownloading account {account.slug}")
        client = ImmichClient(config.base_url, account.api_key)
        user = client.get_my_user()
        owner_id = str(user.get("id", ""))
        album = client.find_album(BIRD_ALBUM_NAME, owner_id=owner_id, owned_only=not args.allow_shared_album)
        if not album:
            print(f"No owned album named {BIRD_ALBUM_NAME!r} found for {account.slug}; skipping")
            continue

        album_id = str(album["id"])
        account_dir = args.output_dir / account.slug
        account_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.manifest_dir / f"{account.slug}_downloaded.csv"
        downloaded_ids = set() if args.overwrite else load_manifest_ids(manifest_path)

        assets = client.iter_album_assets(album_id, page_size=args.page_size, limit=args.limit)
        for asset in progress(assets, desc=account.slug, unit="asset"):
            asset_id = str(asset["id"])
            original_name = str(asset.get("originalFileName") or Path(str(asset.get("originalPath", ""))).name)
            filename = safe_filename(f"{asset_id}_{original_name}", fallback=f"{asset_id}.jpg")
            destination = account_dir / filename

            if not args.overwrite and (asset_id in downloaded_ids or destination.exists()):
                continue

            temp_path = destination.with_suffix(destination.suffix + ".download")
            headers = client.download_original(asset_id, temp_path)
            header_filename = filename_from_headers(headers)
            if header_filename:
                filename = safe_filename(f"{asset_id}_{header_filename}", fallback=filename)
                destination = account_dir / filename

            temp_path.replace(destination)
            append_csv(
                manifest_path,
                manifest_fields,
                {
                    "account": account.slug,
                    "album_id": album_id,
                    "asset_id": asset_id,
                    "original_file_name": original_name,
                    "local_path": destination,
                    "downloaded_at": utc_now(),
                },
            )
            downloaded_ids.add(asset_id)


if __name__ == "__main__":
    main()
