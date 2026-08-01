from __future__ import annotations

from lib.telegram.stream import (
    STREAM_CURSOR,
    StreamingCaptions,
    StreamingMessage,
    StreamRegistry,
)


class FakeNotifier:
    def __init__(self, *, send_ok: bool = True, edit_ok: bool = True) -> None:
        self.sent: list[tuple[int, str]] = []
        self.edits: list[tuple[int, int, str]] = []
        self.caption_edits: list[tuple[int, int, str]] = []
        self._send_ok = send_ok
        self._edit_ok = edit_ok
        self._next_id = 100

    def send_text(self, chat_id, text):
        self.sent.append((chat_id, text))
        if not self._send_ok:
            return None
        self._next_id += 1
        return self._next_id

    def edit_message_text(self, chat_id, message_id, text):
        self.edits.append((chat_id, message_id, text))
        return self._edit_ok

    def edit_message_caption(self, chat_id, message_id, caption):
        self.caption_edits.append((chat_id, message_id, caption))
        return self._edit_ok


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_streaming_message_sends_first_partial_then_edits() -> None:
    notifier = FakeNotifier()
    clock = FakeClock()
    stream = StreamingMessage(notifier, 7, min_interval=1.5, clock=clock)

    stream.update("Percy")
    assert notifier.sent == [(7, "Percy" + STREAM_CURSOR)]
    assert stream.started

    clock.now += 2.0
    stream.update("Percy is")
    assert notifier.edits == [(7, 101, "Percy is" + STREAM_CURSOR)]

    stream.finalize("Percy is napping.")
    assert notifier.edits[-1] == (7, 101, "Percy is napping.")


def test_streaming_message_throttles_edits() -> None:
    notifier = FakeNotifier()
    clock = FakeClock()
    stream = StreamingMessage(notifier, 7, min_interval=1.5, clock=clock)

    stream.update("a")
    clock.now += 0.5
    stream.update("ab")   # inside the interval — skipped
    clock.now += 0.5
    stream.update("abc")  # still inside — skipped
    clock.now += 1.0
    stream.update("abcd")
    assert [t for _, _, t in notifier.edits] == ["abcd" + STREAM_CURSOR]
    # The skipped partials are folded into the final text regardless.
    stream.finalize("abcde")
    assert notifier.edits[-1][2] == "abcde"


def test_streaming_message_finalize_without_start_sends_once() -> None:
    notifier = FakeNotifier()
    stream = StreamingMessage(notifier, 7)
    stream.finalize("Hello!")
    assert notifier.sent == [(7, "Hello!")]
    assert notifier.edits == []


def test_streaming_message_finalize_empty_keeps_partial_visible() -> None:
    notifier = FakeNotifier()
    clock = FakeClock()
    stream = StreamingMessage(notifier, 7, clock=clock)
    stream.update("Percy is")
    stream.finalize("")  # superseded with nothing better — drop the cursor only
    assert notifier.edits[-1][2] == "Percy is"


def test_streaming_message_finalize_falls_back_to_send_when_edit_fails() -> None:
    notifier = FakeNotifier(edit_ok=False)
    clock = FakeClock()
    stream = StreamingMessage(notifier, 7, clock=clock)
    stream.update("part")
    # The first update sent (not edited); make the throttle window pass and push
    # one failing edit so shown != final.
    stream.finalize("the whole answer")
    assert notifier.sent[-1] == (7, "the whole answer")


def test_stream_registry_supersedes_previous_stream() -> None:
    registry = StreamRegistry()
    first = registry.begin(42)
    assert not first.is_set()
    second = registry.begin(42)
    assert first.is_set()      # older stream told to stop
    assert not second.is_set()

    registry.cancel(42)
    assert second.is_set()

    registry.finish(42, second)
    third = registry.begin(42)
    assert not third.is_set()


def test_stream_registry_finish_ignores_stale_handle() -> None:
    registry = StreamRegistry()
    first = registry.begin(1)
    second = registry.begin(1)
    registry.finish(1, first)  # stale — must not evict the newer stream
    registry.cancel(1)
    assert second.is_set()


def test_streaming_captions_edits_all_recipients_with_header() -> None:
    notifier = FakeNotifier()
    clock = FakeClock()
    captions = StreamingCaptions(
        notifier, {"u1": 5, "u2": 9}, header="🐦 10:00 — Percy", clock=clock
    )
    captions.update("Percy preened")
    assert len(notifier.caption_edits) == 2
    assert notifier.caption_edits[0][2] == "🐦 10:00 — Percy\nPercy preened" + STREAM_CURSOR

    clock.now += 10.0
    captions.finalize("Percy preened by the window.")
    assert notifier.caption_edits[-1][2] == "🐦 10:00 — Percy\nPercy preened by the window."


def test_streaming_captions_first_update_is_immediate_then_throttled() -> None:
    notifier = FakeNotifier()
    clock = FakeClock()
    clock.now = 0.0
    captions = StreamingCaptions(notifier, {"u1": 5}, min_interval=3.0, clock=clock)
    captions.update("a")          # last_edit starts at 0.0 — the clock says 0.0
    clock.now = 3.5
    captions.update("ab")
    clock.now = 4.0
    captions.update("abc")        # throttled
    assert [t for _, _, t in notifier.caption_edits] == ["ab" + STREAM_CURSOR]
    captions.finalize("abc")
    assert notifier.caption_edits[-1][2] == "abc"


def test_streaming_captions_finalize_empty_without_updates_is_noop() -> None:
    notifier = FakeNotifier()
    captions = StreamingCaptions(notifier, {"u1": 5}, header="🐦 header")
    captions.finalize("")
    assert notifier.caption_edits == []


def test_streaming_captions_text_mode_uses_edit_message_text() -> None:
    notifier = FakeNotifier()
    clock = FakeClock()
    captions = StreamingCaptions(notifier, {"u1": 5}, captions=False, clock=clock)
    clock.now += 10.0
    captions.finalize("body")
    assert notifier.edits == [("u1", 5, "body")]
    assert notifier.caption_edits == []
