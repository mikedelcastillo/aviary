"""Tests for aviary_immich.console — logic + the plain-print (no-rich) fallback.

We do NOT assert rich rendering internals. For the plain path we force the fallback by
monkeypatching ``console.get_console`` to return ``None`` (or by flipping ``console._RICH``).
"""

from __future__ import annotations

import types

import pytest

import aviary_immich.console as console


# --------------------------------------------------------------------------- helpers


def _force_plain(monkeypatch):
    """Force the plain-print branch by making get_console() return None."""
    monkeypatch.setattr(console, "get_console", lambda: None)


# --------------------------------------------------------------------------- grand_total_table


def test_grand_total_empty_list_no_output(monkeypatch, capsys):
    _force_plain(monkeypatch)
    console.grand_total_table([])
    captured = capsys.readouterr()
    assert captured.out == ""


def test_grand_total_empty_returns_early_even_with_rich(capsys):
    # With rich available get_console() would normally print; empty input must return early.
    console.grand_total_table([])
    captured = capsys.readouterr()
    assert captured.out == ""


def test_grand_total_sums_counts(monkeypatch, capsys):
    _force_plain(monkeypatch)
    per_account = [
        {
            "slug": "acct-a",
            "scanned": 10,
            "Birds": 3,
            "Dogs": 1,
            "Cats": 2,
            "other": 4,
            "errors": 1,
            "added": 3,
            "elapsed": 1.5,
        },
        {
            "slug": "acct-b",
            "scanned": 5,
            "Birds": 2,
            "Dogs": 0,
            "Cats": 1,
            "other": 1,
            "errors": 2,
            "added": 2,
            "elapsed": 2.25,
        },
    ]
    console.grand_total_table(per_account)
    out = capsys.readouterr().out
    assert "Grand total" in out
    assert "scanned=15" in out
    assert "Birds=5" in out
    assert "Dogs=1" in out
    assert "Cats=3" in out
    assert "other=5" in out
    assert "errors=3" in out
    assert "added=5" in out


def test_grand_total_elapsed_is_float_sum(monkeypatch, capsys):
    _force_plain(monkeypatch)
    per_account = [
        {"scanned": 1, "elapsed": 1.5},
        {"scanned": 1, "elapsed": 2.25},
    ]
    console.grand_total_table(per_account)
    out = capsys.readouterr().out
    # 1.5 + 2.25 = 3.75 — a float, not an int.
    assert "elapsed=3.75" in out


def test_grand_total_missing_keys_default_to_zero(monkeypatch, capsys):
    _force_plain(monkeypatch)
    # Entries with no count keys at all -> totals default to 0 / 0.0.
    console.grand_total_table([{"slug": "x"}, {"slug": "y"}])
    out = capsys.readouterr().out
    assert "scanned=0" in out
    assert "Birds=0" in out
    assert "added=0" in out
    assert "elapsed=0.0" in out


def test_grand_total_coerces_string_counts(monkeypatch, capsys):
    _force_plain(monkeypatch)
    # int(...) coercion of string-valued counts.
    console.grand_total_table([{"scanned": "4", "Birds": "2"}])
    out = capsys.readouterr().out
    assert "scanned=4" in out
    assert "Birds=2" in out


# --------------------------------------------------------------------------- account_summary_table


def test_account_summary_plain_contains_slug_and_labels(monkeypatch, capsys):
    _force_plain(monkeypatch)
    stats = {
        "scanned": 7,
        "already": 1,
        "Birds": 2,
        "Dogs": 0,
        "Cats": 1,
        "other": 3,
        "errors": 0,
        "added": 2,
        "elapsed": 4.2,
    }
    console.account_summary_table("my-acct", stats)
    out = capsys.readouterr().out
    assert "my-acct" in out
    assert "scanned" in out
    assert "scanned=7" in out
    assert "Birds=2" in out
    assert "added to album=2" in out


def test_account_summary_elapsed_formatted_one_decimal(monkeypatch, capsys):
    _force_plain(monkeypatch)
    console.account_summary_table("slug", {"elapsed": 3.456})
    out = capsys.readouterr().out
    assert "elapsed=3.5s" in out


def test_account_summary_empty_stats_defaults(monkeypatch, capsys):
    _force_plain(monkeypatch)
    console.account_summary_table("empty", {})
    out = capsys.readouterr().out
    assert "empty" in out
    assert "scanned=0" in out
    assert "elapsed=0.0s" in out


def test_account_summary_plain_no_error(monkeypatch):
    # Plain branch must complete without raising.
    _force_plain(monkeypatch)
    console.account_summary_table("ok", {"scanned": 1})


# --------------------------------------------------------------------------- config_panel


def _make_args(model="yolo.pt", threshold=0.5):
    return types.SimpleNamespace(model=model, threshold=threshold)


def test_config_panel_plain_contains_key_values(monkeypatch, capsys):
    _force_plain(monkeypatch)
    args = _make_args(model="birds.pt", threshold=0.25)
    console.config_panel(
        args,
        device="cuda:0",
        worker_count=4,
        chunk_size=32,
        inference_batch_size=64,
        download_workers=8,
        prefetch=2,
        cpu_workers=1,
    )
    out = capsys.readouterr().out
    assert "Detector config" in out
    assert "model=birds.pt" in out
    assert "device=cuda:0" in out
    assert "threshold=0.25" in out
    assert "workers=4" in out
    assert "cpu workers=1" in out
    assert "download workers=8" in out
    assert "chunk size=32" in out
    assert "inference batch=64" in out
    assert "prefetch=2" in out


def test_config_panel_default_cpu_workers(monkeypatch, capsys):
    _force_plain(monkeypatch)
    console.config_panel(
        _make_args(),
        device="cpu",
        worker_count=2,
        chunk_size=16,
        inference_batch_size=8,
        download_workers=4,
        prefetch=1,
    )
    out = capsys.readouterr().out
    assert "device=cpu" in out
    assert "cpu workers=0" in out


def test_config_panel_plain_no_error(monkeypatch):
    _force_plain(monkeypatch)
    console.config_panel(
        _make_args(),
        device="cpu",
        worker_count=1,
        chunk_size=1,
        inference_batch_size=1,
        download_workers=1,
        prefetch=0,
    )


# --------------------------------------------------------------------------- get_console


def test_get_console_returns_none_when_rich_absent(monkeypatch):
    monkeypatch.setattr(console, "_RICH", False)
    monkeypatch.setattr(console, "_CONSOLE", None)
    assert console.get_console() is None


@pytest.mark.skipif(not console._RICH, reason="rich not installed in this environment")
def test_get_console_caches_when_rich_available(monkeypatch):
    monkeypatch.setattr(console, "_RICH", True)
    monkeypatch.setattr(console, "_CONSOLE", None)
    first = console.get_console()
    second = console.get_console()
    assert first is not None
    assert first is second


@pytest.mark.skipif(not console._RICH, reason="rich not installed in this environment")
def test_get_console_reuses_existing_cached_console(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(console, "_RICH", True)
    monkeypatch.setattr(console, "_CONSOLE", sentinel)
    # Already-cached console is returned as-is without constructing a new one.
    assert console.get_console() is sentinel
