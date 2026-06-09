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
from contextlib import ExitStack
from dataclasses import dataclass, field
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
    parser.add_argument("--cpu-workers", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-rescan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-only", action="store_true", help="Force CPU; ignore any GPU.")
    mode.add_argument(
        "--gpu-only",
        action="store_true",
        help="Force GPU with no hybrid CPU workers; error if no GPU is found.",
    )
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


def detected_labels(record: dict[str, Any], known: set[str]) -> set[str]:
    """Return the animal categories matched by a record, intersected with ``known``.

    Reads the ``labels`` field written by new scans; falls back to the per-detection labels so
    state files written before multi-category support (which only carry ``detections``) still
    route correctly.
    """
    labels = record.get("labels")
    if labels is None:
        labels = [detection.get("label") for detection in record.get("detections", [])]
    return {str(label).lower() for label in labels if label} & known


def category_confidence(record: dict[str, Any], label: str) -> float:
    """Highest confidence among detections of ``label`` (falls back to the record's overall max)."""
    confidences = [
        float(detection.get("confidence") or 0)
        for detection in record.get("detections", [])
        if str(detection.get("label", "")).lower() == label
    ]
    return max(confidences) if confidences else float(record.get("max_confidence") or 0)


def bump_decision(stats: dict[str, int], record: dict[str, Any]) -> None:
    from aviary_immich.config import ANIMAL_LABELS

    if str(record.get("decision")) == "error":
        stats["errors"] = stats.get("errors", 0) + 1
        return
    labels = detected_labels(record, set(ANIMAL_LABELS))
    if not labels:
        stats["other"] = stats.get("other", 0) + 1
        return
    for label in labels:
        key = f"{label}s"
        stats[key] = stats.get(key, 0) + 1


def scan_postfix(stats: dict[str, int], prefix: str = "") -> str:
    return (
        f"{prefix}birds={stats.get('birds', 0)} dogs={stats.get('dogs', 0)} "
        f"cats={stats.get('cats', 0)} err={stats.get('errors', 0)}"
    )


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
    labels: tuple[str, ...],
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
    _WORKER_DETECTOR = PretrainedBirdDetector(model, threshold, device, labels)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER_CACHE_DIR = Path(cache_dir)
    _WORKER_ACCOUNT_SLUG = account_slug
    _WORKER_THUMBNAIL_SIZE = thumbnail_size


def scan_asset_worker(asset: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CLIENT is None or _WORKER_DETECTOR is None or _WORKER_CACHE_DIR is None:
        raise RuntimeError("Scan worker was not initialized")

    try:
        thumbnail_path = thumbnail_for_asset(
            _WORKER_CLIENT,
            _WORKER_CACHE_DIR,
            _WORKER_ACCOUNT_SLUG,
            str(asset["id"]),
            _WORKER_THUMBNAIL_SIZE,
        )
        prediction = _WORKER_DETECTOR.predict(thumbnail_path)
        return record_from_prediction(asset, _WORKER_ACCOUNT_SLUG, prediction)
    except Exception as exc:
        return record_from_error(asset, _WORKER_ACCOUNT_SLUG, exc)


def scan_asset_worker_with_detector(
    asset: dict[str, Any],
    client,
    detector,
    cache_dir: Path,
    account_slug: str,
    thumbnail_size: str,
) -> dict[str, Any]:
    try:
        thumbnail_path = thumbnail_for_asset(client, cache_dir, account_slug, str(asset["id"]), thumbnail_size)
        prediction = detector.predict(thumbnail_path)
        return record_from_prediction(asset, account_slug, prediction)
    except Exception as exc:
        return record_from_error(asset, account_slug, exc)


def record_from_prediction(asset: dict[str, Any], account_slug: str, prediction) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    labels = sorted({str(detection.get("label", "")).lower() for detection in prediction.detections})
    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "match" if labels else "not_match",
        "labels": labels,
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


def _is_cuda_oom(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def predict_batch_adaptive(
    detector,
    assets: list[dict[str, Any]],
    paths: list[Path],
    account_slug: str,
) -> list[dict[str, Any]]:
    """Run one batch, halving and retrying on CUDA OOM so small-VRAM GPUs still run.

    The first time a size OOMs we remember a smaller cap on the detector (``_infer_cap``) and
    pre-split future batches to it, so we don't repeatedly attempt (and fail) the large size.
    A single image that still OOMs, or any non-OOM failure, is recorded as a per-asset error.
    """
    cap = getattr(detector, "_infer_cap", None)
    if cap is not None and len(paths) > cap:
        records: list[dict[str, Any]] = []
        for index in range(0, len(paths), cap):
            records.extend(
                predict_batch_adaptive(detector, assets[index : index + cap], paths[index : index + cap], account_slug)
            )
        return records

    oom = False
    try:
        predictions = detector.predict_batch(paths, batch_size=len(paths))
        if len(predictions) != len(assets):
            raise RuntimeError(f"Detector returned {len(predictions)} results for {len(assets)} images")
        return [record_from_prediction(asset, account_slug, prediction) for asset, prediction in zip(assets, predictions)]
    except Exception as exc:  # noqa: BLE001 - recorded per asset, not fatal
        if _is_cuda_oom(exc) and len(paths) > 1:
            oom = True
        else:
            return [record_from_error(asset, account_slug, exc) for asset in assets]

    # The except block has exited, so the exception (and the traceback that pinned the failed
    # forward pass's GPU tensors) is gone — only now can empty_cache actually reclaim it.
    _free_cuda()
    detector._infer_cap = max(1, len(paths) // 2)
    mid = len(paths) // 2
    return predict_batch_adaptive(detector, assets[:mid], paths[:mid], account_slug) + predict_batch_adaptive(
        detector, assets[mid:], paths[mid:], account_slug
    )


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
        records.extend(predict_batch_adaptive(detector, assets, paths, account_slug))
        if progress_advance is not None:
            progress_advance(len(assets))

    return records


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


def run_gpu_pipeline(
    scan_assets: list[dict[str, Any]],
    detector,
    client,
    base_url: str,
    api_key: str,
    account,
    targets: list[AlbumTarget],
    args: argparse.Namespace,
    chunk_size: int,
    inference_batch_size: int,
    state: dict[str, dict[str, Any]],
    state_appender,
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
                bump_decision(stats, record)
                state_appender.write(record)
                state[str(record["asset_id"])] = record
                handle_scan_record(
                    record,
                    assets_by_id.get(str(record["asset_id"])),
                    account,
                    client,
                    targets,
                    args.dry_run,
                    args.batch_size,
                    stats,
                )
            if progress is not None:
                progress.update(tasks["detect"], postfix=scan_postfix(stats))

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
            tasks["detect"] = progress.add_task("detect", total=total, postfix=scan_postfix(stats))
            run()


def drain_queue_iter(work_queue: queue.Queue, stop_event: threading.Event | None = None) -> Iterable[Any]:
    """Yield items from ``work_queue`` until it is empty (or ``stop_event`` is set)."""
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            yield work_queue.get_nowait()
        except queue.Empty:
            return


def run_hybrid_pipeline(
    scan_assets: list[dict[str, Any]],
    detector,
    client,
    base_url: str,
    api_key: str,
    account,
    targets: list[AlbumTarget],
    args: argparse.Namespace,
    inference_batch_size: int,
    cpu_workers: int,
    state: dict[str, dict[str, Any]],
    state_appender,
    stats: dict[str, int],
) -> None:
    """Saturate the GPU and CPU at once: both steal from one shared asset queue.

    A single GPU thread (the detector is not thread-safe) pulls batches and runs batched
    inference, while a pool of CPU worker processes drains the same queue one asset at a
    time. The faster device naturally takes more work. All finished records flow through a
    single result thread that is the sole owner of the (non-thread-safe) state/album writes.
    """
    from aviary_immich.console import make_scan_progress

    assets_by_id = {str(asset["id"]): asset for asset in scan_assets}
    client_factory = thread_local_client_factory(base_url, api_key)

    asset_queue: queue.Queue = queue.Queue()
    for asset in scan_assets:
        asset_queue.put(asset)
    results_queue: queue.Queue = queue.Queue()
    # Bounded so the downloader runs a few batches ahead of the GPU (overlap) without using
    # unbounded memory. This is what keeps the GPU busy instead of idling on each download.
    gpu_feed_queue: queue.Queue = queue.Queue(maxsize=max(2, args.prefetch))
    gpu_producer_done = threading.Event()
    stop_event = threading.Event()
    error_sink: dict[str, BaseException] = {}
    # Each device thread is the only writer of its own counter, so plain ints are safe.
    device_counts = {"gpu": 0, "cpu": 0}

    progress = make_scan_progress()
    total = len(scan_assets)
    tasks: dict[str, Any] = {"scan": None}

    def gpu_producer() -> None:
        """Download thumbnails ahead of the GPU so inference never waits on the network.

        Steals in inference-batch units (not big chunks) so the CPU pool keeps getting a
        fair share of the shared queue.
        """
        try:
            while not stop_event.is_set() and "gpu_consumer" not in error_sink:
                batch: list[dict[str, Any]] = []
                for _ in range(inference_batch_size):
                    try:
                        batch.append(asset_queue.get_nowait())
                    except queue.Empty:
                        break
                if not batch:
                    break
                cached, error_records = cache_thumbnail_chunk(
                    batch,
                    client_factory,
                    args.cache_dir,
                    account.slug,
                    args.thumbnail_size,
                    args.download_workers,
                )
                while not stop_event.is_set() and "gpu_consumer" not in error_sink:
                    try:
                        gpu_feed_queue.put((cached, error_records), timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            error_sink["gpu_producer"] = exc
        finally:
            gpu_producer_done.set()

    def gpu_consumer() -> None:
        """Run batched GPU inference on downloaded thumbnails (the only GPU thread)."""
        try:
            while True:
                try:
                    cached, error_records = gpu_feed_queue.get(timeout=0.25)
                except queue.Empty:
                    if gpu_producer_done.is_set():
                        break
                    continue
                for record in error_records:
                    results_queue.put(record)
                records = scan_asset_batch_with_detector(
                    cached, detector, account.slug, inference_batch_size
                )
                for record in records:
                    results_queue.put(record)
                device_counts["gpu"] += len(error_records) + len(records)
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            error_sink["gpu_consumer"] = exc

    def cpu_loop(pool) -> None:
        try:
            for record in pool.imap_unordered(
                scan_asset_worker, drain_queue_iter(asset_queue, stop_event)
            ):
                results_queue.put(record)
                device_counts["cpu"] += 1
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            error_sink["cpu"] = exc

    def result_loop(consumer_threads: list[threading.Thread]) -> None:
        handled = 0
        while handled < total:
            try:
                record = results_queue.get(timeout=0.25)
            except queue.Empty:
                # Watchdog: if both producers are gone and nothing is queued, a record was
                # lost (e.g. a worker died) — stop rather than block forever.
                if all(not thread.is_alive() for thread in consumer_threads) and results_queue.empty():
                    break
                continue
            bump_decision(stats, record)
            state_appender.write(record)
            state[str(record["asset_id"])] = record
            handle_scan_record(
                record,
                assets_by_id.get(str(record["asset_id"])),
                account,
                client,
                targets,
                args.dry_run,
                args.batch_size,
                stats,
            )
            handled += 1
            if progress is not None:
                progress.update(
                    tasks["scan"],
                    advance=1,
                    postfix=scan_postfix(stats, f"gpu={device_counts['gpu']} cpu={device_counts['cpu']} "),
                )

    def run() -> None:
        pool = mp.get_context("spawn").Pool(
            processes=cpu_workers,
            initializer=init_scan_worker,
            initargs=(
                base_url,
                api_key,
                args.model,
                args.threshold,
                "cpu",
                tuple(target.label for target in targets),
                str(args.cache_dir),
                account.slug,
                args.thumbnail_size,
            ),
        )
        threads = [
            threading.Thread(target=gpu_producer, name=f"{account.slug}-gpu-dl", daemon=True),
            threading.Thread(target=gpu_consumer, name=f"{account.slug}-gpu", daemon=True),
            threading.Thread(target=cpu_loop, args=(pool,), name=f"{account.slug}-cpu", daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            result_loop(threads)
        finally:
            stop_event.set()
            # Let consumers wind down naturally before terminating the pool, so a normal
            # finish doesn't surface a spurious "pool terminated" error from imap.
            for thread in threads:
                thread.join(timeout=10.0)
            pool.terminate()
            pool.join()
        for key in ("gpu_producer", "gpu_consumer", "cpu"):
            if key in error_sink:
                raise error_sink[key]

    if progress is None:
        run()
    else:
        with progress:
            tasks["scan"] = progress.add_task("hybrid", total=total, postfix=scan_postfix(stats, "gpu=0 cpu=0 "))
            run()


def main() -> None:
    args = parse_args()

    from aviary_immich.client import ImmichClient
    from aviary_immich.config import ANIMAL_ALBUMS, ANIMAL_LABELS, load_accounts_config
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
        load_jsonl_state,
        load_manifest_ids,
    )

    config = load_accounts_config(args.accounts_config, args.env_file)
    if args.cpu_only:
        selected_device = "cpu"
    elif args.gpu_only:
        selected_device = select_device(args.device)
        if selected_device == "cpu":
            emit("[red]--gpu-only set but no GPU was detected.[/]")
            raise SystemExit(1)
    else:
        selected_device = select_device(args.device)
    worker_count = max(1, args.workers)
    chunk_size = max(1, args.chunk_size)
    inference_batch_size = max(1, args.inference_batch_size)
    cpu_workers = 0
    if selected_device != "cpu":
        worker_count = 1
        if args.gpu_only:
            cpu_workers = 0  # pure GPU: run_gpu_pipeline only, no hybrid CPU workers
        else:
            cpu_count = os.cpu_count() or 1
            # Downloads now share cores with CPU inference, so don't oversubscribe them.
            hybrid_download_workers = min(args.download_workers, max(2, cpu_count // 2))
            # Auto: leave the GPU pipeline (download threads + host-side preprocessing) and the
            # result/orchestration thread enough cores so the GPU never starves; CPU gets the rest.
            if args.cpu_workers >= 0:
                cpu_workers = args.cpu_workers
            else:
                cpu_workers = max(0, cpu_count - hybrid_download_workers - 2)
            if cpu_workers > 0:
                args.download_workers = hybrid_download_workers
    config_panel(
        args,
        selected_device,
        worker_count,
        chunk_size,
        inference_batch_size,
        args.download_workers,
        args.prefetch,
        cpu_workers,
    )
    if cpu_workers >= 4:
        emit(
            f"[yellow]Hybrid mode: {cpu_workers} CPU workers each load a full copy of "
            f"{args.model} into RAM — lower --cpu-workers if memory is tight.[/]"
        )
    detector = None
    if worker_count == 1:
        detector = PretrainedBirdDetector(args.model, args.threshold, selected_device, ANIMAL_LABELS)
        if selected_device != "cpu":
            emit(f"[dim]device {selected_device} — fp16 {'on' if detector.half else 'off'}[/]")
    emit(f"[dim]classifying: {', '.join(ANIMAL_ALBUMS.values())}[/]")

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

        state_path = args.state_dir / f"{account.slug}_scan.jsonl"
        state = load_jsonl_state(state_path)

        stats: dict[str, Any] = {
            "scanned": 0,
            "already": 0,
            "birds": 0,
            "dogs": 0,
            "cats": 0,
            "other": 0,
            "errors": 0,
            "added": 0,
            "elapsed": 0.0,
        }

        with ExitStack() as stack:
            # One JSONL state file per account (a single scan, multi-label records); one CSV
            # manifest per album. All scan paths share these open appenders.
            state_appender = stack.enter_context(JsonlAppender(state_path))

            targets: list[AlbumTarget] = []
            for label, album_name in ANIMAL_ALBUMS.items():
                album = {"id": "dry-run", "albumName": album_name}
                if not args.dry_run:
                    album = client.ensure_owned_album(album_name)
                manifest_path = args.manifest_dir / f"{account.slug}_{album_name.lower()}.csv"
                manifested_ids = load_manifest_ids(manifest_path)
                appender = stack.enter_context(CsvAppender(manifest_path, manifest_fields))
                targets.append(
                    AlbumTarget(
                        label=label,
                        album_name=str(album.get("albumName") or album_name),
                        album_id=str(album["id"]),
                        album_ids=set() if args.dry_run else album_asset_ids(client, str(album["id"]), args.page_size),
                        manifested_ids=manifested_ids,
                        write_manifest=appender.write,
                    )
                )

            account_header(
                account.slug,
                connected_as,
                ", ".join(target.album_name for target in targets),
                None,
                None,
                args.dry_run,
            )
            if not args.dry_run:
                for target in targets:
                    emit(
                        f"  [bold]{target.album_name}[/] ([dim]{target.album_id}[/]) — "
                        f"{len(target.album_ids)} assets already present"
                    )

            assets = list(progress(client.iter_image_assets(page_size=args.page_size, limit=args.limit), desc=account.slug, unit="asset"))
            scan_assets: list[dict[str, Any]] = []
            for asset in assets:
                record = state.get(str(asset["id"]))
                if record and not args.force_rescan:
                    handle_scan_record(record, asset, account, client, targets, args.dry_run, args.batch_size, stats)
                else:
                    scan_assets.append(asset)
            stats["scanned"] = len(scan_assets)
            stats["already"] = len(assets) - len(scan_assets)

            if scan_assets:
                if selected_device != "cpu":
                    if cpu_workers > 0:
                        run_hybrid_pipeline(
                            scan_assets,
                            detector,
                            client,
                            config.base_url,
                            account.api_key,
                            account,
                            targets,
                            args,
                            inference_batch_size,
                            cpu_workers,
                            state,
                            state_appender,
                            stats,
                        )
                    else:
                        run_gpu_pipeline(
                            scan_assets,
                            detector,
                            client,
                            config.base_url,
                            account.api_key,
                            account,
                            targets,
                            args,
                            chunk_size,
                            inference_batch_size,
                            state,
                            state_appender,
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
                        state_appender.write(record)
                        state[str(record["asset_id"])] = record
                        bump_decision(stats, record)
                        handle_scan_record(record, asset, account, client, targets, args.dry_run, args.batch_size, stats)
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
                            ANIMAL_LABELS,
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
                            state_appender.write(record)
                            state[str(record["asset_id"])] = record
                            bump_decision(stats, record)
                            handle_scan_record(
                                record,
                                assets_by_id.get(str(record["asset_id"])),
                                account,
                                client,
                                targets,
                                args.dry_run,
                                args.batch_size,
                                stats,
                            )

            for target in targets:
                flush_album_batch(client, target.album_id, target.pending, args.dry_run)
                target.album_ids.update(target.pending_ids)
                target.pending_ids.clear()

        stats["elapsed"] = time.perf_counter() - started
        account_summary_table(account.slug, stats)
        per_account.append({"slug": account.slug, **stats})

    grand_total_table(per_account)


if __name__ == "__main__":
    main()
