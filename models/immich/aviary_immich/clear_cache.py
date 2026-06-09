"""Implementation behind the ``clear-cache`` console script: full reset of generate-albums' data.

All of generate-albums' throwaway and resumable data lives under one directory
(``models/immich/data`` by default), with three subdirs: ``cache/`` (downloaded preview
thumbnails), ``manifests/`` (per-album CSV logs), and ``state/`` (per-account scan JSONL — the
record of what has been processed). ``clear-cache`` wipes **all three** (a full reset): the next
``generate-albums`` run then re-downloads and re-scans everything from scratch. Each subdir's
``.gitkeep`` is preserved so the git-tracked structure stays in place.

Kept dependency-free (stdlib only) so it imports cheaply and is unit-testable.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aviary_immich.ui import emit

# The throwaway/resumable subdirs under the data dir, cleared in this order.
DATA_SUBDIRS = ("cache", "manifests", "state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete generate-albums' cache, manifests, and scan state (full reset)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("models/immich/data"),
        help="generate-albums data directory holding cache/, manifests/, and state/.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting.")
    return parser.parse_args()


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def clear_cache(cache_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Delete everything under ``cache_dir`` except ``.gitkeep``. Returns ``(files, bytes_freed)``.

    Preserving ``.gitkeep`` keeps the git-tracked empty directory in place so the next run still has
    somewhere to cache into.
    """
    if not cache_dir.exists():
        return (0, 0)

    files = 0
    total = 0
    for entry in sorted(cache_dir.iterdir()):
        if entry.name == ".gitkeep":
            continue
        if entry.is_dir():
            for found in entry.rglob("*"):
                if found.is_file():
                    files += 1
                    total += found.stat().st_size
            if not dry_run:
                shutil.rmtree(entry)
        else:
            files += 1
            total += entry.stat().st_size
            if not dry_run:
                entry.unlink()
    return files, total


def clear_data(data_dir: Path, dry_run: bool = False) -> dict[str, tuple[int, int]]:
    """Clear each throwaway/resumable subdir under ``data_dir`` (``cache``/``manifests``/``state``).

    Returns ``{subdir: (files, bytes_freed)}`` for the subdirs that exist. Each is cleared with
    :func:`clear_cache`, so its ``.gitkeep`` is preserved and the directory stays in place.
    """
    results: dict[str, tuple[int, int]] = {}
    for sub in DATA_SUBDIRS:
        target = data_dir / sub
        if target.exists():
            results[sub] = clear_cache(target, dry_run=dry_run)
    return results


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        emit(f"[dim]Nothing to clear: {args.data_dir} does not exist.[/]")
        return
    results = clear_data(args.data_dir, dry_run=args.dry_run)
    if not results:
        emit(f"[dim]Nothing to clear under {args.data_dir} (no cache/, manifests/, or state/).[/]")
        return
    verb = "Would delete" if args.dry_run else "Deleted"
    total_files = 0
    total_bytes = 0
    for sub in DATA_SUBDIRS:
        if sub not in results:
            continue
        files, freed = results[sub]
        total_files += files
        total_bytes += freed
        emit(f"  [dim]{sub}[/]: {files} file(s) ({_human(freed)})")
    emit(f"{verb} {total_files} file(s) ({_human(total_bytes)}) from [bold]{args.data_dir}[/]")


if __name__ == "__main__":
    main()
