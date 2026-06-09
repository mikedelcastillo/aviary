"""Album targets and the fan-out that files each scan record into its per-category albums."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aviary_immich.records import asset_name, category_confidence, detected_labels
from aviary_immich.ui import emit


@dataclass
class AlbumTarget:
    """One animal category and the per-album state needed to file photos into its album."""

    label: str
    album_name: str
    album_id: str
    album_ids: set[str]
    manifested_ids: set[str]
    write_manifest: Callable[[dict[str, Any]], None]
    pending: list[str] = field(default_factory=list)
    pending_ids: set[str] = field(default_factory=set)


def album_asset_ids(client, album_id: str, page_size: int) -> set[str]:
    return {str(asset["id"]) for asset in client.iter_album_assets(album_id, page_size=page_size)}


def flush_album_batch(client, album_id: str, pending: list[str], dry_run: bool) -> int:
    if not pending:
        return 0
    count = len(pending)
    if dry_run:
        emit(f"DRY RUN: would add {count} assets to album {album_id}")
    else:
        client.add_assets_to_album(album_id, pending)
        emit(f"Added {count} assets to album {album_id}")
    pending.clear()
    return count


def handle_scan_record(
    record: dict[str, Any],
    asset: dict[str, Any] | None,
    account,
    client,
    targets: list[AlbumTarget],
    dry_run: bool,
    batch_size: int,
    stats: dict[str, int] | None = None,
) -> None:
    """Fan a scan record out to every album whose category it matched."""
    from aviary_immich.state import utc_now

    asset_id = str(record["asset_id"])
    original_file_name = str(record.get("original_file_name") or (asset_name(asset) if asset else ""))
    labels = detected_labels(record, {target.label for target in targets})
    if not labels:
        return

    for target in targets:
        if target.label not in labels:
            continue

        if asset_id not in target.manifested_ids:
            target.write_manifest(
                {
                    "account": account.slug,
                    "asset_id": asset_id,
                    "decision": target.label,
                    "max_confidence": f"{category_confidence(record, target.label):.4f}",
                    "original_file_name": original_file_name,
                    "album_name": target.album_name,
                    "created_at": utc_now(),
                }
            )
            target.manifested_ids.add(asset_id)

        if asset_id in target.album_ids or asset_id in target.pending_ids:
            continue

        target.pending.append(asset_id)
        target.pending_ids.add(asset_id)
        if stats is not None:
            stats["added"] = stats.get("added", 0) + 1
        if len(target.pending) >= batch_size:
            flush_album_batch(client, target.album_id, target.pending, dry_run)
            target.album_ids.update(target.pending_ids)
            target.pending_ids.clear()
