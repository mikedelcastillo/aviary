from __future__ import annotations

import threading

from lib.find import (
    BirdFinder,
    currently_visible,
    format_found_message,
    format_not_found_message,
    format_progress_message,
    format_visible,
    short_camera,
)


class FakeRegistry:
    """Stands in for ObjectRegistry: returns canned snapshot rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def snapshot(self) -> list[dict]:
        return list(self._rows)


def row(camera: str, label: str, since: float) -> dict:
    return {"camera": camera, "label": label, "since": since}


# -- pure helpers -----------------------------------------------------------


def test_short_camera_trims_to_last_octet() -> None:
    assert short_camera("camera-192.168.1.8") == ".8"
    assert short_camera("studio-cam") == "studio-cam"


def test_currently_visible_filters_by_freshness() -> None:
    rows = [
        row("camera-192.168.1.8", "percy", 2.0),
        row("camera-192.168.1.42", "percy", 30.0),  # stale -> excluded
        row("camera-192.168.1.42", "matcha", 1.0),
    ]
    visible = currently_visible(rows, fresh_seconds=12.0)
    assert visible == {
        "percy": ["camera-192.168.1.8"],
        "matcha": ["camera-192.168.1.42"],
    }


def test_currently_visible_dedupes_and_sorts_cameras() -> None:
    rows = [
        row("camera-192.168.1.44", "matcha", 1.0),
        row("camera-192.168.1.8", "matcha", 1.0),
        row("camera-192.168.1.8", "matcha", 2.0),
    ]
    visible = currently_visible(rows, fresh_seconds=12.0)
    # Cameras are de-duplicated and sorted lexicographically by full name.
    assert visible == {"matcha": ["camera-192.168.1.44", "camera-192.168.1.8"]}


def test_format_visible_and_messages() -> None:
    assert format_visible({}) == "no birds in view"
    visible = {"percy": ["camera-192.168.1.8"], "matcha": ["camera-192.168.1.42"]}
    rendered = format_visible(visible)
    assert "percy (.8)" in rendered
    assert "matcha (.42)" in rendered
    assert "Found percy" in format_found_message("percy", ["camera-192.168.1.8"])
    assert "Still looking for percy" in format_progress_message("percy", visible)
    not_found = format_not_found_message("percy", elapsed=300.0, ever_seen=visible)
    assert "Couldn't find percy" in not_found
    assert "5 min" in not_found
    assert "matcha" in not_found  # recap of what was seen


# -- validation -------------------------------------------------------------


def _finder(registry, sent, **kwargs):
    return BirdFinder(
        registry,
        lambda: ["percy", "matcha", "lovebird"],
        notify=lambda chat_id, text: sent.append((chat_id, text)),
        clock=kwargs.pop("clock", lambda: 0.0),
        poll_seconds=0.0,
        **kwargs,
    )


def test_normalise_target_is_case_insensitive() -> None:
    finder = _finder(FakeRegistry([]), [])
    assert finder.normalise_target("Percy") == "percy"
    assert finder.normalise_target("  MATCHA ") == "matcha"
    assert finder.normalise_target("dog") is None
    assert finder.normalise_target("") is None


def test_start_unknown_bird_lists_options() -> None:
    finder = _finder(FakeRegistry([]), [])
    message = finder.start(123, "dog", threading.Event())
    assert "don't know a bird" in message
    assert "percy" in message


def test_start_without_target_returns_usage() -> None:
    finder = _finder(FakeRegistry([]), [])
    message = finder.start(123, "", threading.Event())
    assert message.startswith("Usage:")
    assert "percy" in message


def test_start_refuses_second_concurrent_search() -> None:
    finder = _finder(FakeRegistry([]), [])
    # Simulate a search already in flight without spawning a thread.
    finder._active_target = "matcha"
    message = finder.start(123, "percy", threading.Event())
    assert "Already searching for matcha" in message


# -- search loop ------------------------------------------------------------


def test_run_reports_when_target_is_visible() -> None:
    sent: list[tuple[int, str]] = []
    registry = FakeRegistry([row("camera-192.168.1.8", "percy", 1.0)])
    finder = _finder(registry, sent, timeout_seconds=300.0)
    outcome = finder._run(123, "percy", threading.Event())
    assert outcome.found is True
    assert outcome.cameras == ["camera-192.168.1.8"]
    assert any("Found percy" in text for _, text in sent)


def test_run_times_out_when_target_absent() -> None:
    sent: list[tuple[int, str]] = []
    registry = FakeRegistry([row("camera-192.168.1.42", "matcha", 1.0)])
    # timeout_seconds=0 -> deadline == start, so the loop falls straight through
    # to the not-found report without sleeping.
    finder = _finder(registry, sent, timeout_seconds=0.0)
    outcome = finder._run(123, "percy", threading.Event())
    assert outcome.found is False
    assert any("Couldn't find percy" in text for _, text in sent)


def test_run_stops_early_when_stop_event_set() -> None:
    sent: list[tuple[int, str]] = []
    registry = FakeRegistry([])
    finder = _finder(registry, sent, timeout_seconds=300.0)
    stop = threading.Event()
    stop.set()
    outcome = finder._run(123, "percy", stop)
    # Stop requested before the first poll: ends as not-found, no crash.
    assert outcome.found is False
