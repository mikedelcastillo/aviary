"""Stream LLM output into Telegram messages via incremental edits.

Telegram has no token streaming, but it does have ``editMessageText`` — the
same mechanism the ``/discover`` progress bar and the caretaker's activity
refresh already use. So a streamed reply is: send the first tokens as soon as
they exist (that's the time-to-first-character win), then grow the message with
throttled edits while the model generates, and finish with one unthrottled
edit of the final cleaned text.

Two shapes are covered:

- :class:`StreamingMessage` — a single chat reply (chat turns, recall answers).
- :class:`StreamingCaptions` — the caretaker report, whose text lives in the
  caption of an already-sent album, per recipient.

:class:`StreamRegistry` keeps at most ONE in-flight streamed reply per chat and
lets a newer user message supersede the old stream immediately (the router
calls :meth:`StreamRegistry.cancel` the moment any new message arrives).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

# Telegram tolerates roughly one edit per second per chat before flood-limiting
# kicks in; editing any faster only buys 429 pauses, not smoother streaming.
STREAM_EDIT_INTERVAL_SECONDS = 1.5
# Caption edits fan out to every recipient per tick, so pace them a bit slower.
CAPTION_EDIT_INTERVAL_SECONDS = 3.0
# Trailing cursor shown while the text is still growing.
STREAM_CURSOR = " ▍"
# Telegram hard limits: message text 4096 chars, media caption 1024.
MESSAGE_LIMIT = 4096
CAPTION_LIMIT = 1024


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class StreamRegistry:
    """At most one in-flight streamed reply per chat, newest wins.

    ``begin`` hands back a cancel :class:`threading.Event` for the new stream
    and sets the previous stream's event so it stops editing its message.
    ``cancel`` is called on every incoming message (see the router) so a stream
    dies the instant the user says something new, not when the reply to the new
    message eventually starts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, threading.Event] = {}

    def begin(self, chat_id: int) -> threading.Event:
        with self._lock:
            previous = self._active.get(chat_id)
            if previous is not None:
                previous.set()
            event = threading.Event()
            self._active[chat_id] = event
            return event

    def cancel(self, chat_id: int) -> None:
        with self._lock:
            event = self._active.get(chat_id)
        if event is not None:
            event.set()

    def finish(self, chat_id: int, event: threading.Event) -> None:
        """Drop the registration, but only if it still belongs to this stream."""
        with self._lock:
            if self._active.get(chat_id) is event:
                del self._active[chat_id]


class StreamingMessage:
    """Send a chat message once, then grow it in place with throttled edits.

    ``update`` is cheap to call per token chunk: the first non-empty partial
    becomes a real send (the user sees text immediately); later partials edit
    that message at most every ``min_interval`` seconds. ``finalize`` bypasses
    the throttle so the last, cleaned text always lands — and falls back to a
    fresh send if the edit fails, so the final answer is never lost.
    """

    def __init__(
        self,
        notifier,
        chat_id: int,
        *,
        min_interval: float = STREAM_EDIT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._notifier = notifier
        self._chat_id = chat_id
        self._min_interval = min_interval
        self._clock = clock
        self._message_id: int | None = None
        self._last_edit = 0.0
        self._shown = ""

    @property
    def started(self) -> bool:
        """Whether any text has reached the user yet."""
        return self._message_id is not None

    @property
    def shown(self) -> str:
        """The last partial the user can currently see (without the cursor)."""
        return self._shown

    def update(self, text: str) -> None:
        text = text.strip()
        if not text or text == self._shown:
            return
        now = self._clock()
        display = _clip(text + STREAM_CURSOR, MESSAGE_LIMIT)
        if self._message_id is None:
            message_id = self._notifier.send_text(self._chat_id, display)
            if message_id is not None:
                self._message_id = message_id
                self._shown = text
                self._last_edit = now
            return
        if now - self._last_edit < self._min_interval:
            return
        if self._notifier.edit_message_text(self._chat_id, self._message_id, display):
            self._shown = text
            self._last_edit = now

    def finalize(self, text: str) -> None:
        text = _clip(text.strip(), MESSAGE_LIMIT)
        if self._message_id is None:
            if text:
                self._notifier.send_text(self._chat_id, text)
            return
        # Never blank an already-sent message; at worst drop the cursor.
        final = text or self._shown
        if not self._notifier.edit_message_text(self._chat_id, self._message_id, final):
            if final != self._shown:
                self._notifier.send_text(self._chat_id, final)


class StreamedReply:
    """A registry-tracked :class:`StreamingMessage` handed to reply code.

    Bundles the cancel event from the registry with the message so a consumer
    (the recall responder) only sees ``update`` / ``cancelled`` / ``finalize``.
    ``finalize`` also releases the registry slot, so calling it is mandatory on
    every path — with ``""`` when there is nothing (more) to say.
    """

    def __init__(self, registry: StreamRegistry, notifier, chat_id: int) -> None:
        self._registry = registry
        self._chat_id = chat_id
        self._event = registry.begin(chat_id)
        self._message = StreamingMessage(notifier, chat_id)
        self.update = self._message.update
        self.cancelled = self._event.is_set

    @property
    def started(self) -> bool:
        return self._message.started

    @property
    def shown(self) -> str:
        return self._message.shown

    def finalize(self, text: str) -> None:
        self._registry.finish(self._chat_id, self._event)
        self._message.finalize(text)


class StreamingCaptions:
    """Grow the caption (or text) of already-sent report messages in place.

    The caretaker report sends its album immediately with a header-only caption
    and then streams the LLM summary into that caption for every recipient —
    the photos arrive without waiting on the language model at all.
    """

    def __init__(
        self,
        notifier,
        messages: dict[str, int],
        *,
        header: str = "",
        captions: bool = True,
        min_interval: float = CAPTION_EDIT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._notifier = notifier
        self._messages = dict(messages)
        self._header = header
        self._captions = captions
        self._limit = CAPTION_LIMIT if captions else MESSAGE_LIMIT
        self._min_interval = min_interval
        self._clock = clock
        self._last_edit = 0.0
        self._shown = ""

    def _compose(self, body: str, *, cursor: bool) -> str:
        text = f"{self._header}\n{body}".strip() if self._header else body.strip()
        if cursor:
            text += STREAM_CURSOR
        return _clip(text, self._limit)

    def _edit_all(self, text: str) -> None:
        for user_id, message_id in self._messages.items():
            if self._captions:
                self._notifier.edit_message_caption(user_id, message_id, text)
            else:
                self._notifier.edit_message_text(user_id, message_id, text)

    def update(self, body: str) -> None:
        body = body.strip()
        if not body or body == self._shown:
            return
        now = self._clock()
        if now - self._last_edit < self._min_interval:
            return
        self._edit_all(self._compose(body, cursor=True))
        self._shown = body
        self._last_edit = now

    def finalize(self, body: str) -> None:
        body = body.strip()
        if not body and not self._shown:
            return  # header-only caption is already what's on screen
        self._edit_all(self._compose(body or self._shown, cursor=False))
        self._shown = body or self._shown
