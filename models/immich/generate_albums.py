#!/usr/bin/env python3
"""Generate per-account Immich Birds albums using a pretrained bird detector."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

_WORKER_CLIENT: Any = None
_WORKER_DETECTOR: Any = None
_WORKER_CACHE_DIR: Path | None = None
_WORKER_ACCOUNT_SLUG = ""
_WORKER_THUMBNAIL_SIZE = "preview"


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
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--download-workers", type=int, default=min(16, (os.cpu_count() or 4) * 2))
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=int(os.getenv("IMMICH_BIRD_WORKERS", str(os.cpu_count() or 1))))
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


def progress_bar(**kwargs):
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(**kwargs)


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


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def album_asset_ids(client, album_id: str, page_size: int) -> set[str]:
    return {str(asset["id"]) for asset in client.iter_album_assets(album_id, page_size=page_size)}


def emit(message: str) -> None:
    """Print a line that renders cleanly above an active rich progress display."""
    from aviary_immich.console import get_console

    console = get_console()
    if console is None:
        print(message)
    else:
        console.log(message)


def bump_decision(stats: dict[str, int], decision: str) -> None:
    if decision == "bird":
        stats["birds"] += 1
    elif decision == "error":
        stats["errors"] += 1
    else:
        stats["other"] += 1


def thread_local_client_factory(base_url: str, api_key: str) -> Callable[[], Any]:
    """Return a factory that hands each thread its own (non-thread-safe) ImmichClient."""
    local = threading.local()

    def factory() -> Any:
        client = getattr(local, "client", None)
        if client is None:
            from aviary_immich.client import ImmichClient

            client = ImmichClient(base_url, api_key)
            local.client = client
        return client

    return factory


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


def init_scan_worker(
    base_url: str,
    api_key: str,
    model: str,
    threshold: float,
    device: str,
    cache_dir: str,
    account_slug: str,
    thumbnail_size: str,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    from aviary_immich.client import ImmichClient
    from aviary_immich.detector import PretrainedBirdDetector

    global _WORKER_CLIENT, _WORKER_DETECTOR, _WORKER_CACHE_DIR, _WORKER_ACCOUNT_SLUG, _WORKER_THUMBNAIL_SIZE
    _WORKER_CLIENT = ImmichClient(base_url, api_key)
    _WORKER_DETECTOR = PretrainedBirdDetector(model, threshold, device)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER_CACHE_DIR = Path(cache_dir)
    _WORKER_ACCOUNT_SLUG = account_slug
    _WORKER_THUMBNAIL_SIZE = thumbnail_size


def scan_asset_worker(asset: dict[str, Any]) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    if _WORKER_CLIENT is None or _WORKER_DETECTOR is None or _WORKER_CACHE_DIR is None:
        raise RuntimeError("Scan worker was not initialized")

    asset_id = str(asset["id"])
    try:
        thumbnail_path = thumbnail_for_asset(
            _WORKER_CLIENT,
            _WORKER_CACHE_DIR,
            _WORKER_ACCOUNT_SLUG,
            asset_id,
            _WORKER_THUMBNAIL_SIZE,
        )
        prediction = _WORKER_DETECTOR.predict(thumbnail_path)
        decision = "bird" if prediction.has_bird else "not_bird"
        return {
            "account": _WORKER_ACCOUNT_SLUG,
            "asset_id": asset_id,
            "decision": decision,
            "max_confidence": prediction.max_confidence,
            "detections": prediction.detections,
            "original_file_name": asset_name(asset),
            "scanned_at": utc_now(),
        }
    except Exception as exc:
        return {
            "account": _WORKER_ACCOUNT_SLUG,
            "asset_id": asset_id,
            "decision": "error",
            "error": str(exc),
            "original_file_name": asset_name(asset),
            "scanned_at": utc_now(),
        }


def scan_asset_worker_with_detector(
    asset: dict[str, Any],
    client,
    detector,
    cache_dir: Path,
    account_slug: str,
    thumbnail_size: str,
) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    asset_id = str(asset["id"])
    try:
        thumbnail_path = thumbnail_for_asset(client, cache_dir, account_slug, asset_id, thumbnail_size)
        prediction = detector.predict(thumbnail_path)
        decision = "bird" if prediction.has_bird else "not_bird"
        return {
            "account": account_slug,
            "asset_id": asset_id,
            "decision": decision,
            "max_confidence": prediction.max_confidence,
            "detections": prediction.detections,
            "original_file_name": asset_name(asset),
            "scanned_at": utc_now(),
        }
    except Exception as exc:
        return {
            "account": account_slug,
            "asset_id": asset_id,
            "decision": "error",
            "error": str(exc),
            "original_file_name": asset_name(asset),
            "scanned_at": utc_now(),
        }


def record_from_prediction(asset: dict[str, Any], account_slug: str, prediction) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    decision = "bird" if prediction.has_bird else "not_bird"
    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": decision,
        "max_confidence": prediction.max_confidence,
        "detections": prediction.detections,
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def record_from_error(asset: dict[str, Any], account_slug: str, exc: Exception) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "error",
        "error": str(exc),
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def cache_thumbnail_chunk(
    assets: list[dict[str, Any]],
    client_factory: Callable[[], Any],
    cache_dir: Path,
    account_slug: str,
    thumbnail_size: str,
    download_workers: int = 1,
    progress_advance: Callable[[int], None] | None = None,
) -> tuple[list[tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    cached: list[tuple[dict[str, Any], Path]] = []
    errors: list[dict[str, Any]] = []

    def fetch(asset: dict[str, Any]):
        client = client_factory()
        try:
            thumbnail_path = thumbnail_for_asset(client, cache_dir, account_slug, str(asset["id"]), thumbnail_size)
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


def scan_asset_batch_with_detector(
    assets_and_paths: list[tuple[dict[str, Any], Path]],
    detector,
    account_slug: str,
    inference_batch_size: int,
    progress_advance: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for batch in chunks(assets_and_paths, inference_batch_size):
        assets = [asset for asset, _ in batch]
        paths = [path for _, path in batch]
        try:
            predictions = detector.predict_batch(paths, batch_size=inference_batch_size)
        except Exception as exc:
            records.extend(record_from_error(asset, account_slug, exc) for asset in assets)
            if progress_advance is not None:
                progress_advance(len(assets))
            continue

        if len(predictions) != len(assets):
            exc = RuntimeError(f"Detector returned {len(predictions)} results for {len(assets)} images")
            records.extend(record_from_error(asset, account_slug, exc) for asset in assets)
            if progress_advance is not None:
                progress_advance(len(assets))
            continue

        records.extend(
            record_from_prediction(asset, account_slug, prediction)
            for asset, prediction in zip(assets, predictions)
        )
        if progress_advance is not None:
            progress_advance(len(assets))

    return records


def handle_scan_record(
    record: dict[str, Any],
    asset: dict[str, Any] | None,
    account,
    client,
    album_id: str,
    dry_run: bool,
    batch_size: int,
    write_manifest: Callable[[dict[str, Any]], None],
    manifested_ids: set[str],
    album_ids: set[str],
    pending: list[str],
    pending_ids: set[str],
    stats: dict[str, int] | None = None,
) -> None:
    from aviary_immich.config import BIRD_ALBUM_NAME
    from aviary_immich.state import utc_now

    asset_id = str(record["asset_id"])
    decision = str(record.get("decision", ""))
    max_confidence = float(record.get("max_confidence") or 0)
    original_file_name = str(record.get("original_file_name") or (asset_name(asset) if asset else ""))

    if decision != "bird":
        return

    if asset_id not in manifested_ids:
        write_manifest(
            {
                "account": account.slug,
                "asset_id": asset_id,
                "decision": decision,
                "max_confidence": f"{max_confidence:.4f}",
                "original_file_name": original_file_name,
                "album_name": BIRD_ALBUM_NAME,
                "created_at": utc_now(),
            }
        )
        manifested_ids.add(asset_id)

    if asset_id in album_ids or asset_id in pending_ids:
        return

    pending.append(asset_id)
    pending_ids.add(asset_id)
    if stats is not None:
        stats["added"] = stats.get("added", 0) + 1
    if len(pending) >= batch_size:
        flush_album_batch(client, album_id, pending, dry_run)
        album_ids.update(pending_ids)
        pending_ids.clear()


def run_gpu_pipeline(
    scan_assets: list[dict[str, Any]],
    detector,
    client,
    base_url: str,
    api_key: str,
    account,
    album_id: str,
    args: argparse.Namespace,
    chunk_size: int,
    inference_batch_size: int,
    state: dict[str, dict[str, Any]],
    state_appender,
    write_manifest: Callable[[dict[str, Any]], None],
    manifested_ids: set[str],
    album_ids: set[str],
    pending: list[str],
    pending_ids: set[str],
    stats: dict[str, int],
) -> None:
    """Overlap thumbnail downloads (producer) with GPU inference (consumer)."""
    from aviary_immich.console import make_scan_progress

    assets_by_id = {str(asset["id"]): asset for asset in scan_assets}
    client_factory = thread_local_client_factory(base_url, api_key)
    work_queue: queue.Queue = queue.Queue(maxsize=max(1, args.prefetch))
    sentinel = object()
    producer_error: dict[str, BaseException] = {}

    progress = make_scan_progress()
    total = len(scan_assets)
    tasks: dict[str, Any] = {"cache": None, "detect": None}

    def cache_advance(n: int = 1) -> None:
        if progress is not None:
            progress.update(tasks["cache"], advance=n)

    def detect_advance(n: int = 1) -> None:
        if progress is not None:
            progress.update(tasks["detect"], advance=n)

    def produce() -> None:
        try:
            for chunk in chunks(scan_assets, chunk_size):
                cached, error_records = cache_thumbnail_chunk(
                    chunk,
                    client_factory,
                    args.cache_dir,
                    account.slug,
                    args.thumbnail_size,
                    args.download_workers,
                    cache_advance,
                )
                work_queue.put((cached, error_records))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer below
            producer_error["exc"] = exc
        finally:
            work_queue.put(sentinel)

    def consume() -> None:
        while True:
            item = work_queue.get()
            if item is sentinel:
                break
            cached, error_records = item
            if error_records:
                detect_advance(len(error_records))
            records = list(error_records)
            records += scan_asset_batch_with_detector(
                cached, detector, account.slug, inference_batch_size, detect_advance
            )
            for record in records:
                bump_decision(stats, str(record.get("decision", "")))
                state_appender.write(record)
                state[str(record["asset_id"])] = record
                handle_scan_record(
                    record,
                    assets_by_id.get(str(record["asset_id"])),
                    account,
                    client,
                    album_id,
                    args.dry_run,
                    args.batch_size,
                    write_manifest,
                    manifested_ids,
                    album_ids,
                    pending,
                    pending_ids,
                    stats,
                )
            if progress is not None:
                progress.update(tasks["detect"], postfix=f"birds={stats['birds']} err={stats['errors']}")

    def run() -> None:
        producer = threading.Thread(target=produce, name=f"{account.slug}-downloader", daemon=True)
        producer.start()
        try:
            consume()
        finally:
            # If the consumer exited early, drain the queue so the producer can finish.
            while producer.is_alive():
                try:
                    work_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
            producer.join()
        if "exc" in producer_error:
            raise producer_error["exc"]

    if progress is None:
        run()
    else:
        with progress:
            tasks["cache"] = progress.add_task("cache", total=total, postfix="")
            tasks["detect"] = progress.add_task("detect", total=total, postfix="birds=0 err=0")
            run()


def main() -> None:
    args = parse_args()

    from aviary_immich.client import ImmichClient
    from aviary_immich.config import BIRD_ALBUM_NAME, load_accounts_config
    from aviary_immich.console import (
        account_header,
        account_summary_table,
        config_panel,
        grand_total_table,
    )
    from aviary_immich.detector import PretrainedBirdDetector, select_device
    from aviary_immich.state import (
        CsvAppender,
        JsonlAppender,
        append_csv,
        append_jsonl,
        load_jsonl_state,
        load_manifest_ids,
    )

    config = load_accounts_config(args.accounts_config, args.env_file)
    selected_device = select_device(args.device)
    worker_count = max(1, args.workers)
    chunk_size = max(1, args.chunk_size)
    inference_batch_size = max(1, args.inference_batch_size)
    if selected_device != "cpu":
        worker_count = 1
    config_panel(
        args,
        selected_device,
        worker_count,
        chunk_size,
        inference_batch_size,
        args.download_workers,
        args.prefetch,
    )
    detector = None
    if worker_count == 1:
        detector = PretrainedBirdDetector(args.model, args.threshold, selected_device)

    manifest_fields = [
        "account",
        "asset_id",
        "decision",
        "max_confidence",
        "original_file_name",
        "album_name",
        "created_at",
    ]

    per_account: list[dict[str, Any]] = []

    for account in config.accounts:
        started = time.perf_counter()
        client = ImmichClient(config.base_url, account.api_key)
        user = client.get_my_user()
        connected_as = str(user.get("email") or user.get("name") or user.get("id"))

        album = {"id": "dry-run", "albumName": BIRD_ALBUM_NAME}
        if not args.dry_run:
            album = client.ensure_owned_album(BIRD_ALBUM_NAME)

        state_path = args.state_dir / f"{account.slug}_scan.jsonl"
        manifest_path = args.manifest_dir / f"{account.slug}_birds.csv"
        state = load_jsonl_state(state_path)
        manifested_ids = load_manifest_ids(manifest_path)
        album_ids = set() if args.dry_run else album_asset_ids(client, str(album["id"]), args.page_size)

        account_header(
            account.slug,
            connected_as,
            str(album.get("albumName") or BIRD_ALBUM_NAME),
            str(album.get("id")) if not args.dry_run else None,
            len(album_ids) if not args.dry_run else None,
            args.dry_run,
        )

        pending: list[str] = []
        pending_ids: set[str] = set()
        stats: dict[str, Any] = {
            "scanned": 0,
            "already": 0,
            "birds": 0,
            "other": 0,
            "errors": 0,
            "added": 0,
            "elapsed": 0.0,
        }

        def write_manifest_plain(row: dict[str, Any]) -> None:
            append_csv(manifest_path, manifest_fields, row)

        assets = list(progress(client.iter_image_assets(page_size=args.page_size, limit=args.limit), desc=account.slug, unit="asset"))
        scan_assets: list[dict[str, Any]] = []
        for asset in assets:
            asset_id = str(asset["id"])
            record = state.get(asset_id)

            if record and not args.force_rescan:
                handle_scan_record(
                    record,
                    asset,
                    account,
                    client,
                    str(album["id"]),
                    args.dry_run,
                    args.batch_size,
                    write_manifest_plain,
                    manifested_ids,
                    album_ids,
                    pending,
                    pending_ids,
                    stats,
                )
            else:
                scan_assets.append(asset)
        stats["scanned"] = len(scan_assets)
        stats["already"] = len(assets) - len(scan_assets)

        if scan_assets:
            if selected_device != "cpu":
                with JsonlAppender(state_path) as state_appender, CsvAppender(manifest_path, manifest_fields) as manifest_appender:
                    run_gpu_pipeline(
                        scan_assets,
                        detector,
                        client,
                        config.base_url,
                        account.api_key,
                        account,
                        str(album["id"]),
                        args,
                        chunk_size,
                        inference_batch_size,
                        state,
                        state_appender,
                        manifest_appender.write,
                        manifested_ids,
                        album_ids,
                        pending,
                        pending_ids,
                        stats,
                    )
            elif worker_count == 1:
                for asset in progress(scan_assets, desc=f"{account.slug} detect", unit="asset"):
                    record = scan_asset_worker_with_detector(
                        asset,
                        client,
                        detector,
                        args.cache_dir,
                        account.slug,
                        args.thumbnail_size,
                    )
                    append_jsonl(state_path, record)
                    state[str(record["asset_id"])] = record
                    bump_decision(stats, str(record.get("decision", "")))
                    handle_scan_record(
                        record,
                        asset,
                        account,
                        client,
                        str(album["id"]),
                        args.dry_run,
                        args.batch_size,
                        write_manifest_plain,
                        manifested_ids,
                        album_ids,
                        pending,
                        pending_ids,
                        stats,
                    )
            else:
                assets_by_id = {str(asset["id"]): asset for asset in scan_assets}
                with mp.get_context("spawn").Pool(
                    processes=worker_count,
                    initializer=init_scan_worker,
                    initargs=(
                        config.base_url,
                        account.api_key,
                        args.model,
                        args.threshold,
                        selected_device,
                        str(args.cache_dir),
                        account.slug,
                        args.thumbnail_size,
                    ),
                ) as pool:
                    for record in progress(
                        pool.imap_unordered(scan_asset_worker, scan_assets),
                        total=len(scan_assets),
                        desc=f"{account.slug} detect",
                        unit="asset",
                    ):
                        append_jsonl(state_path, record)
                        state[str(record["asset_id"])] = record
                        bump_decision(stats, str(record.get("decision", "")))
                        handle_scan_record(
                            record,
                            assets_by_id.get(str(record["asset_id"])),
                            account,
                            client,
                            str(album["id"]),
                            args.dry_run,
                            args.batch_size,
                            write_manifest_plain,
                            manifested_ids,
                            album_ids,
                            pending,
                            pending_ids,
                            stats,
                        )

        flush_album_batch(client, str(album["id"]), pending, args.dry_run)
        stats["elapsed"] = time.perf_counter() - started
        account_summary_table(account.slug, stats)
        per_account.append({"slug": account.slug, **stats})

    grand_total_table(per_account)


if __name__ == "__main__":
    main()
