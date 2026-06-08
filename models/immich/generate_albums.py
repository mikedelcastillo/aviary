#!/usr/bin/env python3
"""Generate per-account Immich Birds albums using a pretrained bird detector."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    env_args, _ = env_parser.parse_known_args()
    from aviary_immich.config import load_env_file

    load_env_file(env_args.env_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-config", type=Path, default=Path("models/immich/config/accounts.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/immich/cache/thumbnails"))
    parser.add_argument("--state-dir", type=Path, default=Path("models/immich/state"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("models/immich/manifests"))
    parser.add_argument("--model", default=os.getenv("IMMICH_BIRD_MODEL", "yolo11x.pt"))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("IMMICH_BIRD_THRESHOLD", "0.30")))
    parser.add_argument("--device", default=os.getenv("IMMICH_BIRD_DEVICE", "auto"))
    parser.add_argument("--thumbnail-size", default=os.getenv("IMMICH_THUMBNAIL_SIZE", "preview"))
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-rescan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def progress(items: Iterable[Any], **kwargs) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, **kwargs)


def thumbnail_for_asset(client, cache_dir: Path, account_slug: str, asset_id: str, size: str) -> Path:
    from aviary_immich.client import suffix_from_headers

    account_cache = cache_dir / account_slug
    account_cache.mkdir(parents=True, exist_ok=True)

    existing = sorted(account_cache.glob(f"{asset_id}.*"))
    existing = [path for path in existing if path.suffix != ".download"]
    if existing:
        return existing[0]

    temp_path = account_cache / f"{asset_id}.download"
    headers = client.download_thumbnail(asset_id, temp_path, size=size)
    final_path = account_cache / f"{asset_id}{suffix_from_headers(headers)}"
    temp_path.replace(final_path)
    return final_path


def asset_name(asset: dict[str, Any]) -> str:
    return str(asset.get("originalFileName") or asset.get("originalPath") or asset.get("id") or "")


def flush_album_batch(client, album_id: str, pending: list[str], dry_run: bool) -> None:
    if not pending:
        return
    if dry_run:
        print(f"DRY RUN: would add {len(pending)} assets to album {album_id}")
    else:
        client.add_assets_to_album(album_id, pending)
        print(f"Added {len(pending)} assets to album {album_id}")
    pending.clear()


def main() -> None:
    args = parse_args()

    from aviary_immich.client import ImmichClient
    from aviary_immich.config import load_accounts_config
    from aviary_immich.detector import PretrainedBirdDetector
    from aviary_immich.state import append_csv, append_jsonl, load_jsonl_state, load_manifest_ids, utc_now

    config = load_accounts_config(args.accounts_config, args.env_file)
    detector = PretrainedBirdDetector(args.model, args.threshold, args.device)
    print(f"Detector loaded: model={args.model} device={detector.device} threshold={args.threshold}")

    manifest_fields = [
        "account",
        "asset_id",
        "decision",
        "max_confidence",
        "original_file_name",
        "album_name",
        "created_at",
    ]

    for account in config.accounts:
        print(f"\nScanning account {account.slug} ({account.name})")
        client = ImmichClient(config.base_url, account.api_key)
        user = client.get_my_user()
        print(f"Connected as {user.get('email') or user.get('name') or user.get('id')}")

        album = {"id": "dry-run", "albumName": account.album_name}
        if not args.dry_run:
            album = client.ensure_owned_album(account.album_name)
            print(f"Using album {album.get('albumName')} ({album.get('id')})")
        else:
            print(f"DRY RUN: would create/find album {account.album_name}")

        state_path = args.state_dir / f"{account.slug}_scan.jsonl"
        manifest_path = args.manifest_dir / f"{account.slug}_birds.csv"
        state = load_jsonl_state(state_path)
        manifested_ids = load_manifest_ids(manifest_path)
        pending: list[str] = []

        assets = client.iter_image_assets(page_size=args.page_size, limit=args.limit)
        for asset in progress(assets, desc=account.slug, unit="asset"):
            asset_id = str(asset["id"])
            record = state.get(asset_id)

            if record and not args.force_rescan:
                decision = str(record.get("decision", ""))
                max_confidence = float(record.get("max_confidence") or 0)
            else:
                try:
                    thumbnail_path = thumbnail_for_asset(
                        client,
                        args.cache_dir,
                        account.slug,
                        asset_id,
                        args.thumbnail_size,
                    )
                    prediction = detector.predict(thumbnail_path)
                    decision = "bird" if prediction.has_bird else "not_bird"
                    max_confidence = prediction.max_confidence
                    record = {
                        "account": account.slug,
                        "asset_id": asset_id,
                        "decision": decision,
                        "max_confidence": max_confidence,
                        "detections": prediction.detections,
                        "original_file_name": asset_name(asset),
                        "scanned_at": utc_now(),
                    }
                except Exception as exc:
                    decision = "error"
                    max_confidence = 0.0
                    record = {
                        "account": account.slug,
                        "asset_id": asset_id,
                        "decision": decision,
                        "error": str(exc),
                        "original_file_name": asset_name(asset),
                        "scanned_at": utc_now(),
                    }

                append_jsonl(state_path, record)
                state[asset_id] = record

            if decision == "bird":
                pending.append(asset_id)
                if asset_id not in manifested_ids:
                    append_csv(
                        manifest_path,
                        manifest_fields,
                        {
                            "account": account.slug,
                            "asset_id": asset_id,
                            "decision": decision,
                            "max_confidence": f"{max_confidence:.4f}",
                            "original_file_name": asset_name(asset),
                            "album_name": account.album_name,
                            "created_at": utc_now(),
                        },
                    )
                    manifested_ids.add(asset_id)

            if len(pending) >= args.batch_size:
                flush_album_batch(client, str(album["id"]), pending, args.dry_run)

        flush_album_batch(client, str(album["id"]), pending, args.dry_run)


if __name__ == "__main__":
    main()
