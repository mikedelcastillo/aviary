from __future__ import annotations

import threading

from lib.find import (
    BirdFinder,
    currently_visible,
    format_found_message,
    format_not_found_message,
    format_progress_message,
    format_visible,
    pretty,
    short_camera,
)


FINDABLE = ["bambi", "cockatiel", "draft", "jynx", "lovebird", "matcha", "percy", "pizza"]


class FakeRegistry:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def snapshot(self) -> list[dict]:
        return list(self._rows)


def row(camera: str, label: str, since: float) -> dict:
    return {"camera": camera, "label": label, "since": since}


# -- pure helpers -----------------------------------------------------------


def test_pretty_capitalises_bird_names() -> None:
    assert pretty("percy") == "Percy"
    assert pretty("unknown_bird") == "Unknown Bird"


def test_short_camera_trims_to_last_octet() -> None:
    assert short_camera("camera-192.168.1.8") == ".8"


def test_currently_visible_filters_by_freshness() -> None:
    rows = [
        row("camera-192.168.1.8", "percy", 2.0),
        row("camera-192.168.1.42", "percy", 30.0),  # stale
        row("camera-192.168.1.42", "matcha", 1.0),
    ]
    assert currently_visible(rows, 12.0) == {
        "percy": ["camera-192.168.1.8"],
        "matcha": ["camera-192.168.1.42"],
    }


def test_format_visible_capitalises() -> None:
    assert "Percy (.8)" in format_visible({"percy": ["camera-192.168.1.8"]})
    assert format_visible({}) == "no birds in view"


def test_format_found_message_lists_birds_and_description() -> None:
    visible = {"draft": ["camera-192.168.1.8"], "percy": ["camera-192.168.1.42"]}
    msg = format_found_message(["draft", "percy"], visible, "Draft is eating.")
    assert "Draft on .8" in msg
    assert "Percy on .42" in msg
    assert "Draft is eating." in msg


def test_format_progress_and_not_found_capitalise() -> None:
    assert "Cockatiels" in format_progress_message("cockatiels", {})
    nf = format_not_found_message("cockatiels", 300.0, {"matcha": ["camera-192.168.1.8"]})
    assert "Cockatiels" in nf and "Matcha" in nf and "5 min" in nf


# -- finder -----------------------------------------------------------------


def _finder(registry, sent, **kwargs):
    return BirdFinder(
        registry,
        lambda: FINDABLE,
        notify=lambda chat_id, text: sent.append((chat_id, text)),
        clock=kwargs.pop("clock", lambda: 0.0),
        poll_seconds=0.0,
        **kwargs,
    )


def test_resolve_targets_handles_groups() -> None:
    finder = _finder(FakeRegistry([]), [])
    assert set(finder.resolve_targets("cockatiels")) == {"draft", "pizza", "cockatiel"}
    # An individual also resolves to its species outline (so IR feeds match).
    assert finder.resolve_targets("percy") == ["percy", "lovebird"]


def test_start_unknown_lists_options() -> None:
    finder = _finder(FakeRegistry([]), [])
    msg = finder.start(1, "dinosaur", threading.Event())
    assert "don't know" in msg.lower()
    assert "Percy" in msg


def test_start_empty_returns_usage() -> None:
    finder = _finder(FakeRegistry([]), [])
    assert finder.start(1, "", threading.Event()).startswith("Usage:")


def test_run_finds_any_member_of_a_group() -> None:
    sent: list = []
    # Only draft is visible; "find the cockatiels" should still succeed on it.
    registry = FakeRegistry([row("camera-192.168.1.8", "draft", 1.0)])
    finder = _finder(registry, sent, timeout_seconds=300.0)
    targets = finder.resolve_targets("cockatiels")
    outcome = finder._run(7, "cockatiels", targets, threading.Event(), threading.Event())
    assert outcome.found is True
    assert outcome.found_labels == ["draft"]
    assert any("Found Draft on .8" in text for _, text in sent)


def test_run_includes_vlm_description_and_photos_on_hit() -> None:
    sent: list = []
    photos: list = []
    registry = FakeRegistry([row("camera-192.168.1.8", "percy", 1.0)])
    finder = BirdFinder(
        registry,
        lambda: FINDABLE,
        notify=lambda c, t: sent.append((c, t)),
        grab_frame=lambda cam: b"jpeg",
        send_photo=lambda c, img, cap: photos.append((c, img, cap)) or True,
        describe_frame=lambda image: "Percy is preening near the window.",
        clock=lambda: 0.0,
        poll_seconds=0.0,
        timeout_seconds=300.0,
    )
    outcome = finder._run(7, "percy", ["percy"], threading.Event(), threading.Event())
    assert outcome.found is True
    # Photo is sent individually (caption names the bird); the VLM description
    # arrives as a separate follow-up so a slow vision model never blocks it.
    assert photos and "Percy" in photos[0][2]
    assert any("Percy is preening" in text for _, text in sent)


def test_run_times_out_when_absent() -> None:
    sent: list = []
    registry = FakeRegistry([row("camera-192.168.1.42", "matcha", 1.0)])
    finder = _finder(registry, sent, timeout_seconds=0.0)
    outcome = finder._run(7, "percy", ["percy"], threading.Event(), threading.Event())
    assert outcome.found is False
    assert any("Couldn't find Percy" in text for _, text in sent)


def test_run_cancelled_exits_quietly() -> None:
    sent: list = []
    registry = FakeRegistry([])
    finder = _finder(registry, sent, timeout_seconds=300.0)
    cancel = threading.Event()
    cancel.set()  # already cancelled before the loop runs
    outcome = finder._run(7, "percy", ["percy"], threading.Event(), cancel)
    assert outcome.found is False
    # Cancellation is announced by stop_current()/start(), not the loop.
    assert not any("Couldn't find" in text for _, text in sent)


def test_stop_current_without_search() -> None:
    finder = _finder(FakeRegistry([]), [])
    assert "No search is running" in finder.stop_current()


def test_start_stop_word_cancels_active_search() -> None:
    finder = _finder(FakeRegistry([]), [])
    # Pretend a search is active.
    finder._active = {"token": object(), "requested": "percy", "cancel": threading.Event()}
    msg = finder.start(1, "stop", threading.Event())
    assert "Stopped searching for Percy" in msg
    assert finder._active["cancel"].is_set()


def test_start_while_active_replaces() -> None:
    # timeout 0 so the launched search thread exits immediately instead of
    # spinning on the constant test clock.
    finder = _finder(FakeRegistry([]), [], timeout_seconds=0.0)
    old_cancel = threading.Event()
    finder._active = {"token": object(), "requested": "matcha", "cancel": old_cancel}
    msg = finder.start(1, "percy", threading.Event())
    # The previous search is cancelled and the new one announced as a switch.
    assert old_cancel.is_set()
    assert "Switching" in msg


def test_bird_last_seen_picks_most_recent_camera() -> None:
    from lib.find import bird_last_seen
    rows = [
        {"label": "percy", "camera": "cam-a", "since": 300.0},
        {"label": "percy", "camera": "cam-b", "since": 12.0},  # more recent
        {"label": "draft", "camera": "cam-a", "since": 50.0},
    ]
    seen = bird_last_seen(rows)
    assert seen["percy"] == (12.0, "cam-b")
    assert seen["draft"] == (50.0, "cam-a")
