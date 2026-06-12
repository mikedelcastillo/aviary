#!/usr/bin/env python3
"""Download phone photos from Immich "Birds" albums into the annotation pipeline.

For each comma-separated API key in `IMMICH_API_KEY`, finds that user's album
named `Birds` and downloads its image assets (preview quality by default) into
`data/annotation/raw/phone/` as `phone_{date}_{assetId}.jpg`, so they can be
labeled alongside the Tapo camera frames.

The asset UUID in the filename makes the name deterministic, so re-running skips
photos that are already downloaded. Configure the server via `.env`:

    IMMICH_BASE_URL=http://host:2283/api   # must include the /api suffix
    IMMICH_API_KEY=key1,key2               # one key per phone/user
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "annotation" / "raw" / "phone",
        help="Destination folder for downloaded phone photos",
    )
    parser.add_argument(
        "--album",
        default="Birds",
        help="Album name to download from each API key (default: Birds)",
    )
    parser.add_argument(
        "--size",
        default="preview",
        choices=["preview", "thumbnail"],
        help="Immich thumbnail size to download (default: preview)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded; write nothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    base_url = os.environ.get("IMMICH_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise SystemExit("IMMICH_BASE_URL is not set (see .env.example)")

    keys = [k.strip() for k in os.environ.get("IMMICH_API_KEY", "").split(",") if k.strip()]
    if not keys:
        raise SystemExit("IMMICH_API_KEY is not set (see .env.example)")

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    mode = " [yellow](dry run)[/]" if args.dry_run else ""
    console.print(
        f"[bold]Immich import[/] · album [cyan]{args.album}[/] · "
        f"[cyan]{args.size}[/] · {len(keys)} key(s){mode}"
    )
    console.print(f"[dim]{base_url} → {args.output}[/]\n")

    downloaded = 0
    skipped = 0
    errors = 0

    for i, key in enumerate(keys, start=1):
        headers = {"x-api-key": key, "Accept": "application/json"}

        try:
            resp = requests.get(f"{base_url}/albums", headers=headers, timeout=30)
            resp.raise_for_status()
            albums = resp.json()
        except requests.RequestException as exc:
            console.print(f"[red]✗[/] key #{i}: albums error — {exc}")
            errors += 1
            continue

        album = next((a for a in albums if a.get("albumName") == args.album), None)
        if album is None:
            console.print(f"[yellow]⚠[/] key #{i}: no '{args.album}' album")
            continue

        try:
            resp = requests.get(f"{base_url}/albums/{album['id']}", headers=headers, timeout=30)
            resp.raise_for_status()
            assets = [a for a in resp.json().get("assets", []) if a.get("type") == "IMAGE"]
        except requests.RequestException as exc:
            console.print(f"[red]✗[/] key #{i}: album info error — {exc}")
            errors += 1
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"key #{i} · {len(assets)} images", total=len(assets))

            for asset in assets:
                asset_id = asset["id"]
                captured = asset.get("fileCreatedAt") or asset.get("localDateTime") or "unknown"
                date = captured[:10]
                dest = args.output / f"phone_{date}_{asset_id}.jpg"

                if dest.exists():
                    skipped += 1
                    progress.advance(task)
                    continue

                if args.dry_run:
                    progress.console.print(f"[dim][dry-run] → {dest.name}[/]")
                    downloaded += 1
                    progress.advance(task)
                    continue

                try:
                    resp = requests.get(
                        f"{base_url}/assets/{asset_id}/thumbnail",
                        headers={"x-api-key": key},
                        params={"size": args.size},
                        timeout=60,
                    )
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    progress.console.print(f"[red]✗[/] download {asset_id}: {exc}")
                    errors += 1
                    progress.advance(task)
                    continue

                dest.write_bytes(resp.content)
                downloaded += 1
                progress.advance(task)

    verb = "Would download" if args.dry_run else "Downloaded"
    console.print(
        f"\n[bold green]✓[/] {verb} [green]{downloaded}[/], "
        f"skipped [yellow]{skipped}[/] (existing), "
        f"[red]{errors}[/] error(s) across {len(keys)} key(s)."
    )


if __name__ == "__main__":
    main()
