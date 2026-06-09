"""Implementation behind the ``clear-cache`` console script: delete generate-albums' local cache.

The thumbnail cache (``models/immich/cache/thumbnails`` by default) is just downloaded preview
images — the album generator re-fetches whatever it needs — so it is safe to delete to reclaim
disk (it can grow to many GB). Scan state (``state/``) and album manifests (``manifests/``) are
deliberately NOT touched: they are the resumable record of what has been classified and filed, not
a cache, and deleting them would force a full re-scan.

Kept dependency-free (stdlib only) so it imports cheaply and is unit-testable.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aviary_immich.ui import emit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete generate-albums' local thumbnail cache.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/immich/cache/thumbnails"),
        help="Cache directory to clear (matches generate-albums' --cache-dir default).",
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


def main() -> None:
    args = parse_args()
    files, freed = clear_cache(args.cache_dir, dry_run=args.dry_run)
    if not args.cache_dir.exists():
        emit(f"[dim]Nothing to clear: {args.cache_dir} does not exist.[/]")
        return
    verb = "Would delete" if args.dry_run else "Deleted"
    emit(f"{verb} {files} cached file(s) ({_human(freed)}) from [bold]{args.cache_dir}[/]")


if __name__ == "__main__":
    main()
