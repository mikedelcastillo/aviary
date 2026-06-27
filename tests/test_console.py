from __future__ import annotations

from lib.console import ConsoleDispatcher, ConsoleNotifier


def _dispatcher(emitted, calls, on_quit=lambda: None):
    return ConsoleDispatcher(
        emit=emitted.append,
        status_text=lambda: "STATUS BOXES",
        discover_text=lambda: "discovered 2",
        restart_text=lambda: "restarting",
        detection_text=lambda arg: f"detections:{arg}",
        snapshot_text=lambda cid: f"snapshot[{cid}]",
        pause=lambda secs: f"paused {secs}",
        resume=lambda: "resumed",
        find=lambda cid, target: f"find[{cid}]:{target}",
        nl_handle=lambda cid, text: calls.append(("nl", cid, text)),
        parse_duration=lambda text: 600.0 if text == "10m" else None,
        toggle_logs=lambda: "logs toggled",
        on_quit=on_quit,
    )


def test_slash_commands_route_to_providers() -> None:
    emitted: list[str] = []
    _dispatcher(emitted, []).handle("/status")
    assert emitted == ["STATUS BOXES"]

    emitted.clear()
    _dispatcher(emitted, []).handle("/pause 10m")
    assert emitted == ["paused 600.0"]

    emitted.clear()
    _dispatcher(emitted, []).handle("/play")
    assert emitted == ["resumed"]

    emitted.clear()
    _dispatcher(emitted, []).handle("/find the cockatiels")
    assert emitted == ["find[0]:the cockatiels"]

    emitted.clear()
    _dispatcher(emitted, []).handle("/discover")
    assert emitted[-1] == "discovered 2"

    emitted.clear()
    _dispatcher(emitted, []).handle("/restart")
    assert emitted == ["restarting"]

    emitted.clear()
    _dispatcher(emitted, []).handle("/detections percy")
    assert emitted == ["detections:percy"]


def test_plain_language_goes_to_nl_router() -> None:
    emitted: list[str] = []
    calls: list = []
    _dispatcher(emitted, calls).handle("where is percy?")
    assert calls == [("nl", 0, "where is percy?")]


def test_logs_and_quit() -> None:
    emitted: list[str] = []
    quit_called: list[bool] = []
    dispatcher = _dispatcher(emitted, [], on_quit=lambda: quit_called.append(True))
    dispatcher.handle("/logs")
    assert "logs toggled" in emitted
    dispatcher.handle("/quit")
    assert quit_called == [True]


def test_blank_line_is_ignored() -> None:
    emitted: list[str] = []
    calls: list = []
    _dispatcher(emitted, calls).handle("   ")
    assert emitted == [] and calls == []


def test_console_notifier_routes_text_and_summarises_album() -> None:
    out: list[str] = []
    notifier = ConsoleNotifier(out.append)
    notifier.send_text(0, "hello")
    notifier.send_album(0, [(b"a", "x"), (b"b", "y")])
    assert out[0] == "hello"
    assert "2 photo" in out[1]  # album becomes a note, not images
