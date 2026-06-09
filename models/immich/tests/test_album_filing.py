"""Tests for album_filing: AlbumTarget fan-out, manifest/album dedup, batch flush."""

from __future__ import annotations

import types

import pytest

from aviary_immich.album_filing import (
    AlbumTarget,
    album_asset_ids,
    flush_album_batch,
    handle_scan_record,
)
from fakes import FakeImmichClient, make_asset, make_detection, make_record


def make_target(label, *, album_name=None, album_id=None, rows=None):
    """An AlbumTarget wired to capture manifest rows on the given list."""
    rows = [] if rows is None else rows
    return AlbumTarget(
        label=label,
        album_name=album_name or f"{label.title()}s",
        album_id=album_id or f"alb-{label}",
        album_ids=set(),
        manifested_ids=set(),
        write_manifest=rows.append,
    )


@pytest.fixture
def account():
    return types.SimpleNamespace(slug="acct")


@pytest.fixture
def frozen_now(monkeypatch):
    monkeypatch.setattr("aviary_immich.state.utc_now", lambda: "2026-06-09T00:00:00Z")
    return "2026-06-09T00:00:00Z"


# --------------------------------------------------------------------------- fan-out


def test_handle_scan_record_fans_out_to_matching_targets(account, frozen_now):
    client = FakeImmichClient()
    bird = make_target("bird")
    dog = make_target("dog")
    cat = make_target("cat")
    record = make_record(asset_id="a1", labels=["bird", "dog"])

    handle_scan_record(record, None, account, client, [bird, dog, cat], dry_run=False, batch_size=100)

    assert bird.pending == ["a1"]
    assert bird.pending_ids == {"a1"}
    assert dog.pending == ["a1"]
    assert dog.pending_ids == {"a1"}


def test_handle_scan_record_leaves_unmatched_target_untouched(account, frozen_now):
    client = FakeImmichClient()
    bird = make_target("bird")
    dog = make_target("dog")
    cat = make_target("cat")
    record = make_record(asset_id="a1", labels=["bird", "dog"])

    handle_scan_record(record, None, account, client, [bird, dog, cat], dry_run=False, batch_size=100)

    assert cat.pending == []
    assert cat.pending_ids == set()
    assert cat.manifested_ids == set()


def test_handle_scan_record_no_match_returns_early(account, frozen_now):
    client = FakeImmichClient()
    rows = []
    bird = make_target("bird", rows=rows)
    record = make_record(asset_id="a1", labels=[])

    handle_scan_record(record, None, account, client, [bird], dry_run=False, batch_size=100)

    assert bird.pending == []
    assert bird.pending_ids == set()
    assert bird.manifested_ids == set()
    assert rows == []
    assert client.added == []


# --------------------------------------------------------------------------- manifest dedup


def test_manifest_written_once_for_new_asset(account, frozen_now):
    client = FakeImmichClient()
    rows = []
    bird = make_target("bird", rows=rows)
    record = make_record(asset_id="a1", labels=["bird"], max_confidence=0.87654321)

    handle_scan_record(record, None, account, client, [bird], dry_run=False, batch_size=100)

    assert len(rows) == 1
    assert "a1" in bird.manifested_ids
    row = rows[0]
    assert row["decision"] == bird.label
    assert row["max_confidence"] == "0.8765"
    assert row["account"] == "acct"
    assert row["asset_id"] == "a1"
    assert row["album_name"] == bird.album_name
    assert row["created_at"] == frozen_now


def test_manifest_uses_detection_confidence_for_category(account, frozen_now):
    client = FakeImmichClient()
    rows = []
    bird = make_target("bird", rows=rows)
    record = make_record(
        asset_id="a1",
        labels=["bird"],
        detections=[make_detection("bird", confidence=0.5)],
        max_confidence=0.99,
    )

    handle_scan_record(record, None, account, client, [bird], dry_run=False, batch_size=100)

    assert rows[0]["max_confidence"] == "0.5000"


def test_manifest_not_rewritten_for_already_manifested_asset(account, frozen_now):
    client = FakeImmichClient()
    rows = []
    bird = make_target("bird", rows=rows)
    bird.manifested_ids.add("a1")
    record = make_record(asset_id="a1", labels=["bird"])

    handle_scan_record(record, None, account, client, [bird], dry_run=False, batch_size=100)

    assert rows == []
    # The asset still gets queued for the album.
    assert bird.pending == ["a1"]


# --------------------------------------------------------------------------- album dedup


def test_asset_already_in_album_ids_not_requeued(account, frozen_now):
    client = FakeImmichClient()
    bird = make_target("bird")
    bird.album_ids.add("a1")
    record = make_record(asset_id="a1", labels=["bird"])

    handle_scan_record(record, None, account, client, [bird], dry_run=False, batch_size=100)

    assert bird.pending == []
    assert bird.pending_ids == set()


def test_asset_already_in_pending_ids_not_requeued(account, frozen_now):
    client = FakeImmichClient()
    bird = make_target("bird")
    bird.pending.append("a1")
    bird.pending_ids.add("a1")
    record = make_record(asset_id="a1", labels=["bird"])

    handle_scan_record(record, None, account, client, [bird], dry_run=False, batch_size=100)

    assert bird.pending == ["a1"]
    assert bird.pending_ids == {"a1"}


# --------------------------------------------------------------------------- batch flush


def test_batch_flush_triggers_on_full_batch(account, frozen_now):
    client = FakeImmichClient()
    bird = make_target("bird", album_id="alb-bird")
    stats = {}

    handle_scan_record(make_record(asset_id="a1", labels=["bird"]), None, account, client, [bird], dry_run=False, batch_size=2, stats=stats)
    assert client.added == []
    assert bird.pending == ["a1"]
    assert stats["added"] == 1

    handle_scan_record(make_record(asset_id="a2", labels=["bird"]), None, account, client, [bird], dry_run=False, batch_size=2, stats=stats)

    assert client.added == [("alb-bird", ["a1", "a2"])]
    assert bird.album_ids == {"a1", "a2"}
    assert bird.pending == []
    assert bird.pending_ids == set()
    assert stats["added"] == 2


# --------------------------------------------------------------------------- flush_album_batch


def test_flush_empty_pending_returns_zero_no_call():
    client = FakeImmichClient()
    pending = []

    result = flush_album_batch(client, "alb-1", pending, dry_run=False)

    assert result == 0
    assert client.added == []
    assert pending == []


def test_flush_dry_run_clears_pending_without_client_call():
    client = FakeImmichClient()
    pending = ["a1", "a2"]

    result = flush_album_batch(client, "alb-1", pending, dry_run=True)

    assert result == 2
    assert client.added == []
    assert pending == []


def test_flush_real_run_calls_client_once():
    client = FakeImmichClient()
    pending = ["a1", "a2", "a3"]

    result = flush_album_batch(client, "alb-1", pending, dry_run=False)

    assert result == 3
    assert client.added == [("alb-1", ["a1", "a2", "a3"])]
    assert pending == []


# --------------------------------------------------------------------------- album_asset_ids


def test_album_asset_ids_returns_str_id_set():
    client = FakeImmichClient(album_assets=[make_asset(id="x"), make_asset(id="y")])

    assert album_asset_ids(client, "alb-1", page_size=250) == {"x", "y"}


def test_album_asset_ids_empty_album():
    client = FakeImmichClient(album_assets=[])

    assert album_asset_ids(client, "alb-1", page_size=10) == set()


def test_album_asset_ids_coerces_non_str_ids():
    client = FakeImmichClient(album_assets=[make_asset(id=1), make_asset(id=2)])

    assert album_asset_ids(client, "alb-1", page_size=10) == {"1", "2"}
