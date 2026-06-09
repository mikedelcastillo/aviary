"""Multiprocessing worker glue, per-thread clients, and queue draining.

The ``_WORKER_*`` module globals hold each spawned worker process's own client/models. Under
the "spawn" start method every child re-imports this module fresh, then the pool initializer
(:func:`init_scan_worker`) populates the globals in that child; :func:`scan_asset_worker` then
reads them in the same process. Keeping the initializer and worker in one module is what lets
``mp.Pool`` ship them by qualified name and resolve the same globals.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from aviary_immich.records import record_from_error, record_from_outputs
from aviary_immich.thumbnails import thumbnail_for_asset

_WORKER_CLIENT: Any = None
_WORKER_MODELS: Any = None
_WORKER_CACHE_DIR: Path | None = None
_WORKER_ACCOUNT_SLUG = ""
_WORKER_THUMBNAIL_SIZE = "preview"


def init_scan_worker(
    base_url: str,
    api_key: str,
    model_specs: tuple[Any, ...],
    default_model: str,
    default_threshold: float,
    device: str,
    cache_dir: str,
    account_slug: str,
    thumbnail_size: str,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    from aviary_immich.client import ImmichClient
    from aviary_immich.models import build_models

    global _WORKER_CLIENT, _WORKER_MODELS, _WORKER_CACHE_DIR, _WORKER_ACCOUNT_SLUG, _WORKER_THUMBNAIL_SIZE
    _WORKER_CLIENT = ImmichClient(base_url, api_key)
    _WORKER_MODELS = build_models(model_specs, device, default_model, default_threshold)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER_CACHE_DIR = Path(cache_dir)
    _WORKER_ACCOUNT_SLUG = account_slug
    _WORKER_THUMBNAIL_SIZE = thumbnail_size


def scan_asset_worker(asset: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CLIENT is None or _WORKER_MODELS is None or _WORKER_CACHE_DIR is None:
        raise RuntimeError("Scan worker was not initialized")

    try:
        thumbnail_path = thumbnail_for_asset(
            _WORKER_CLIENT,
            _WORKER_CACHE_DIR,
            _WORKER_ACCOUNT_SLUG,
            str(asset["id"]),
            _WORKER_THUMBNAIL_SIZE,
        )
        outputs = {model.name: model.predict_paths([thumbnail_path], batch_size=1)[0] for model in _WORKER_MODELS}
        return record_from_outputs(asset, _WORKER_ACCOUNT_SLUG, outputs)
    except Exception as exc:
        return record_from_error(asset, _WORKER_ACCOUNT_SLUG, exc)


def scan_asset_worker_with_models(
    asset: dict[str, Any],
    client,
    models,
    cache_dir: Path,
    account_slug: str,
    thumbnail_size: str,
) -> dict[str, Any]:
    try:
        thumbnail_path = thumbnail_for_asset(client, cache_dir, account_slug, str(asset["id"]), thumbnail_size)
        outputs = {model.name: model.predict_paths([thumbnail_path], batch_size=1)[0] for model in models}
        return record_from_outputs(asset, account_slug, outputs)
    except Exception as exc:
        return record_from_error(asset, account_slug, exc)


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


def drain_queue_iter(work_queue: queue.Queue, stop_event: threading.Event | None = None) -> Iterable[Any]:
    """Yield items from ``work_queue`` until it is empty (or ``stop_event`` is set)."""
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            yield work_queue.get_nowait()
        except queue.Empty:
            return
