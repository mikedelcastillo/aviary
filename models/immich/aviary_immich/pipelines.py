"""GPU-only and hybrid GPU+CPU scan pipelines.

These orchestrate the perf-critical threading/multiprocessing dance that keeps the GPU saturated.
The logic they call (thumbnail caching, batched inference, OOM halving, record fan-out, album
flushing) lives in the sibling modules and is unit-tested there; the pipelines themselves are
wiring and are exercised end-to-end via the CLI.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import threading
from typing import Any

from aviary_immich.album_filing import AlbumTarget, handle_scan_record
from aviary_immich.config import MODELS
from aviary_immich.inference import scan_asset_batch_with_models
from aviary_immich.records import bump_decision, chunks, scan_postfix
from aviary_immich.thumbnails import cache_thumbnail_chunk, delete_cached_thumbnail
from aviary_immich.workers import (
    drain_queue_iter,
    init_scan_worker,
    scan_asset_worker,
    thread_local_client_factory,
)


def run_gpu_pipeline(
    scan_assets: list[dict[str, Any]],
    models,
    client,
    base_url: str,
    api_key: str,
    account,
    targets: list[AlbumTarget],
    args: argparse.Namespace,
    inference_batch_size: int,
    state: dict[str, dict[str, Any]],
    state_appender,
    stats: dict[str, int],
) -> None:
    """Three-stage pipeline so the GPU thread does nothing but forward passes.

    download+decode workers ──work_queue──▶ one GPU thread ──results_queue──▶ one result thread
       (cv2.imread, GIL-free,   (bounded by      (batched           (unbounded;    (sole owner of
        N threads)               prefetch)        inference only)     tiny dicts)    state + albums)

    Decode moves to the download pool (cv2 releases the GIL) and all per-record bookkeeping —
    state writes plus the blocking ``add_assets_to_album`` PUTs inside ``handle_scan_record`` —
    moves to the result thread, so neither steals time from GPU inference. The producer emits at
    ``inference_batch_size`` granularity (not ``chunk_size``) so decoded arrays in flight stay
    bounded (~``prefetch * inference_batch_size`` images) and the downloader runs ahead of the GPU.
    """
    from aviary_immich.console import make_scan_progress

    assets_by_id = {str(asset["id"]): asset for asset in scan_assets}
    client_factory = thread_local_client_factory(base_url, api_key)
    feed_batch = max(1, inference_batch_size)
    work_queue: queue.Queue = queue.Queue(maxsize=max(1, args.prefetch))
    results_queue: queue.Queue = queue.Queue()
    work_sentinel = object()
    results_sentinel = object()
    errors: dict[str, BaseException] = {}

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
        """Download + decode thumbnails ahead of the GPU, emitting array batches."""
        try:
            for batch in chunks(scan_assets, feed_batch):
                cached, error_records = cache_thumbnail_chunk(
                    batch,
                    client_factory,
                    args.cache_dir,
                    account.slug,
                    args.thumbnail_size,
                    args.download_workers,
                    cache_advance,
                    decode=True,
                )
                work_queue.put((cached, error_records))
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            errors["producer"] = exc
        finally:
            work_queue.put(work_sentinel)

    def consume_gpu() -> None:
        """The only GPU thread: run batched inference on decoded arrays, emit records."""
        try:
            while True:
                item = work_queue.get()
                if item is work_sentinel:
                    break
                cached, error_records = item
                for record in error_records:
                    results_queue.put(record)
                if error_records:
                    detect_advance(len(error_records))
                for record in scan_asset_batch_with_models(
                    cached, models, account.slug, inference_batch_size, detect_advance, use_arrays=True
                ):
                    results_queue.put(record)
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            errors["gpu"] = exc
        finally:
            results_queue.put(results_sentinel)

    def result_loop() -> None:
        """Sole owner of the non-thread-safe state dict, state file, and album writes."""
        while True:
            record = results_queue.get()
            if record is results_sentinel:
                break
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
            # Record is durable in state now (so it won't be re-downloaded next run); the decoded
            # image already drove inference, so its cached thumbnail is dead weight — drop it.
            if not args.keep_cache:
                delete_cached_thumbnail(args.cache_dir, account.slug, str(record["asset_id"]))
            if progress is not None:
                progress.update(tasks["detect"], postfix=scan_postfix(stats))

    def run() -> None:
        producer = threading.Thread(target=produce, name=f"{account.slug}-downloader", daemon=True)
        gpu = threading.Thread(target=consume_gpu, name=f"{account.slug}-gpu", daemon=True)
        producer.start()
        gpu.start()
        try:
            result_loop()
        finally:
            # Drain both queues so a producer/GPU thread blocked on a full queue can still reach
            # its sentinel and exit, even if the result loop aborted early.
            while producer.is_alive() or gpu.is_alive():
                for drained in (work_queue, results_queue):
                    try:
                        while True:
                            drained.get_nowait()
                    except queue.Empty:
                        pass
                producer.join(timeout=0.1)
                gpu.join(timeout=0.1)
        for key in ("producer", "gpu"):
            if key in errors:
                raise errors[key]

    if progress is None:
        run()
    else:
        with progress:
            tasks["cache"] = progress.add_task("cache", total=total, postfix="")
            tasks["detect"] = progress.add_task("detect", total=total, postfix=scan_postfix(stats))
            run()


def run_hybrid_pipeline(
    scan_assets: list[dict[str, Any]],
    models,
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

    A single GPU thread (models are not thread-safe; the GPU thread runs all of them) pulls
    batches and runs batched inference, while a pool of CPU worker processes drains the same
    queue one asset at a time. The faster device naturally takes more work. All finished
    records flow through a single result thread that is the sole owner of the (non-thread-safe)
    state/album writes.
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
                    decode=True,
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
                records = scan_asset_batch_with_models(
                    cached, models, account.slug, inference_batch_size, use_arrays=True
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
            # Durable in state; decoded image already used — drop the cached thumbnail.
            if not args.keep_cache:
                delete_cached_thumbnail(args.cache_dir, account.slug, str(record["asset_id"]))
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
                MODELS,
                args.model,
                args.threshold,
                "cpu",
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
