"""Tests for aviary_immich.workers: per-thread client factory, the in-process
detector worker, queue draining, and picklability of the spawn-shipped callables.

No real multiprocessing Pool is created and no model is loaded; the heavy
collaborators are replaced with the shared fakes and tmp_path.
"""

from __future__ import annotations

import pickle
import queue
import threading

import pytest

from aviary_immich import workers
from fakes import FakeImmichClient, FakeModel, OutOfMemoryError, make_asset, make_output


# --------------------------------------------------------------------------- thread_local_client_factory


class _CountingClient:
    """Stand-in for ImmichClient that counts how many times it is constructed."""

    instances: list["_CountingClient"] = []

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        type(self).instances.append(self)


@pytest.fixture
def counting_client(monkeypatch):
    _CountingClient.instances = []
    # The factory does `from aviary_immich.client import ImmichClient` at call time,
    # so patch the attribute on the client module.
    monkeypatch.setattr("aviary_immich.client.ImmichClient", _CountingClient)
    return _CountingClient


def test_factory_reuses_one_instance_within_same_thread(counting_client):
    factory = workers.thread_local_client_factory("http://host", "key")

    first = factory()
    second = factory()

    assert first is second
    # Exactly one construction despite two calls in the same thread.
    assert len(counting_client.instances) == 1


def test_factory_passes_base_url_and_api_key(counting_client):
    factory = workers.thread_local_client_factory("http://host", "secret")

    client = factory()

    assert client.base_url == "http://host"
    assert client.api_key == "secret"


def test_factory_gives_each_thread_its_own_instance(counting_client):
    factory = workers.thread_local_client_factory("http://host", "key")

    main_client = factory()

    other = {}

    def run():
        # Two calls in the worker thread should still only construct once *for that thread*.
        other["first"] = factory()
        other["second"] = factory()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert other["first"] is other["second"]
    # The second thread got its own distinct instance.
    assert other["first"] is not main_client
    # Two threads -> two constructions total (one per thread).
    assert len(counting_client.instances) == 2


# --------------------------------------------------------------------------- scan_asset_worker_with_models


def test_worker_with_models_returns_match_record(tmp_path):
    asset = make_asset(id="asset-1", name="pic.jpg")
    client = FakeImmichClient()
    model = FakeModel(name="yolo", responder=lambda items: [make_output(labels=["bird"]) for _ in items])

    record = workers.scan_asset_worker_with_models(
        asset, client, [model], tmp_path, "acct", "preview"
    )

    assert record["decision"] == "match"
    assert record["asset_id"] == "asset-1"
    assert record["account"] == "acct"
    assert record["labels"] == ["bird"]
    # Provenance is keyed by the model name.
    assert record["model_tags"] == {"yolo": {"bird": 0.9}}
    # The thumbnail was downloaded through the fake client.
    assert client.downloaded == ["asset-1"]


def test_worker_with_models_uses_predict_single_path(tmp_path):
    asset = make_asset(id="asset-1")
    client = FakeImmichClient()
    model = FakeModel(name="yolo", responder=lambda items: [make_output(labels=["bird"]) for _ in items])

    workers.scan_asset_worker_with_models(asset, client, [model], tmp_path, "acct", "preview")

    # The worker routes through predict_paths with a single path and batch_size=1.
    assert model.calls == [("paths", 1, 1)]


def test_worker_with_models_returns_error_record_on_raise(tmp_path):
    asset = make_asset(id="asset-err")
    client = FakeImmichClient()

    def boom(items):
        raise OutOfMemoryError("CUDA out of memory")

    model = FakeModel(name="yolo", responder=boom)

    record = workers.scan_asset_worker_with_models(
        asset, client, [model], tmp_path, "acct", "preview"
    )

    assert record["decision"] == "error"
    assert record["asset_id"] == "asset-err"
    assert record["account"] == "acct"
    assert "out of memory" in record["error"].lower()
    # No prediction labels on an error record.
    assert "labels" not in record


def test_worker_with_models_no_bird_is_not_match(tmp_path):
    asset = make_asset(id="asset-empty")
    client = FakeImmichClient()
    model = FakeModel(name="yolo", responder=lambda items: [make_output(labels=[]) for _ in items])

    record = workers.scan_asset_worker_with_models(
        asset, client, [model], tmp_path, "acct", "preview"
    )

    assert record["decision"] == "not_match"
    assert record["labels"] == []


# --------------------------------------------------------------------------- drain_queue_iter


def test_drain_queue_yields_all_items_then_stops():
    work_queue: queue.Queue = queue.Queue()
    for item in ("x", "y", "z"):
        work_queue.put(item)

    drained = list(workers.drain_queue_iter(work_queue))

    assert drained == ["x", "y", "z"]
    # Generator stopped cleanly on Empty; queue is now empty.
    assert work_queue.empty()


def test_drain_queue_with_set_stop_event_yields_nothing():
    work_queue: queue.Queue = queue.Queue()
    for item in ("x", "y", "z"):
        work_queue.put(item)

    stop_event = threading.Event()
    stop_event.set()

    drained = list(workers.drain_queue_iter(work_queue, stop_event=stop_event))

    assert drained == []
    # Items were left untouched in the queue.
    assert work_queue.qsize() == 3


def test_drain_queue_empty_queue_yields_nothing():
    work_queue: queue.Queue = queue.Queue()

    assert list(workers.drain_queue_iter(work_queue)) == []


def test_drain_queue_unset_event_drains_normally():
    work_queue: queue.Queue = queue.Queue()
    work_queue.put("only")

    stop_event = threading.Event()  # not set

    assert list(workers.drain_queue_iter(work_queue, stop_event=stop_event)) == ["only"]


# --------------------------------------------------------------------------- picklability smoke


def test_scan_asset_worker_is_picklable_by_qualified_name():
    restored = pickle.loads(pickle.dumps(workers.scan_asset_worker))
    assert restored is workers.scan_asset_worker


def test_init_scan_worker_is_picklable_by_qualified_name():
    restored = pickle.loads(pickle.dumps(workers.init_scan_worker))
    assert restored is workers.init_scan_worker
