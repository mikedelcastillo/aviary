"""Telegram alert delivery."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from lib.detector import Detection


LOGGER = logging.getLogger("lib.telegram")

# Telegram caps a single ``sendMediaGroup`` album at 10 items; larger snapshot
# sets are split into successive albums.
MEDIA_GROUP_LIMIT = 10

# Telegram answers bursts with HTTP 429 and a ``retry_after`` (seconds). Every
# send shares one bot token, so the limit is global: ``_blocked_until`` parks
# every recipient thread when a 429 lands, and ``_next_send_at`` paces normal
# traffic so we stay clear of the limit in the first place.
DEFAULT_MIN_SEND_INTERVAL_SECONDS = 0.05
DEFAULT_MAX_SEND_RETRIES = 5
# Cushion added on top of Telegram's retry_after so we resume just past the
# window instead of racing its trailing edge.
RETRY_AFTER_BUFFER_SECONDS = 1.0
# Fallback pause when a 429 arrives without any retry_after hint.
RETRY_AFTER_FALLBACK_SECONDS = 1.0


def _chunked(
    items: list[tuple[bytes, str | None]], size: int
) -> Iterator[list[tuple[bytes, str | None]]]:
    """Yield successive ``size``-length slices of ``items`` (last may be shorter)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        user_ids: list[str],
        timeout_seconds: int = 15,
        photo_timeout_seconds: int = 60,
        min_send_interval_seconds: float = DEFAULT_MIN_SEND_INTERVAL_SECONDS,
        max_send_retries: int = DEFAULT_MAX_SEND_RETRIES,
    ) -> None:
        self.bot_token = bot_token
        self.user_ids = user_ids
        self.timeout_seconds = timeout_seconds
        # Photo uploads are far heavier than text and slow uplinks need headroom.
        self.photo_timeout_seconds = photo_timeout_seconds
        self.min_send_interval_seconds = min_send_interval_seconds
        self.max_send_retries = max_send_retries
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        # Deliver to all recipients concurrently so 3-4 recipients cost one
        # send's worth of latency, not the sum. Sized to the recipient count.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, len(user_ids)),
            thread_name_prefix="telegram-send",
        )
        # Shared rate-limit gate guarding the two timestamps below.
        self._rate_lock = threading.Lock()
        self._next_send_at = 0.0
        self._blocked_until = 0.0

    # -- public API --------------------------------------------------------

    def send_detections(
        self,
        camera_name: str,
        detections: list[Detection],
        snapshot_path: Path | None,
    ) -> None:
        if not self.bot_token or not self.user_ids:
            raise ValueError("Telegram bot token and user IDs are required")

        text = ", ".join(sorted({detection.label for detection in detections}))
        has_photo = bool(snapshot_path and snapshot_path.exists())

        if not has_photo:
            self._broadcast(self.user_ids, lambda user_id: self._send_message(user_id, text))
            return

        # Upload the image bytes once. Telegram returns a ``file_id`` for the
        # stored photo; every remaining recipient is served that handle instead
        # of re-uploading the bytes, so N recipients cost one upload plus N-1
        # cheap references.
        image_bytes = snapshot_path.read_bytes()
        file_id: str | None = None
        uploaded: set[str] = set()
        for user_id in self.user_ids:
            try:
                file_id = self._send_photo_upload(user_id, text, image_bytes)
                uploaded.add(user_id)
            except requests.RequestException as exc:
                LOGGER.warning("Photo upload to %s failed: %s", user_id, exc)
                continue
            if file_id:
                # Got a reusable handle; stop pushing bytes and fan the rest out.
                break

        if file_id:
            remaining = [user_id for user_id in self.user_ids if user_id not in uploaded]
            if remaining:
                self._broadcast(
                    remaining,
                    lambda user_id: self._send_photo_by_file_id(user_id, text, file_id),
                )
        # If no file_id was ever obtained, the loop above already attempted a
        # direct upload to every recipient, so there is nothing left to send.

    def send_text(self, chat_id: int | str, text: str) -> None:
        """Send a plain text message to a SINGLE chat (used by ``/find`` updates).

        Goes through the shared rate-limit gate like every other send, so a burst
        of search updates paces alongside alerts under the one bot-token limit. A
        transient API error is logged and swallowed — a dropped progress line
        must never tear down the search thread.
        """
        try:
            self._send_message(str(chat_id), text)
        except requests.RequestException as exc:
            LOGGER.warning("Text send to %s failed: %s", chat_id, exc)

    def send_album(
        self, chat_id: int | str, items: Sequence[tuple[bytes, str | None]]
    ) -> None:
        """Send images to a SINGLE chat as album(s) (used by ``/snapshot``).

        ``items`` is ``(image_bytes, caption)`` pairs. Telegram caps a media
        group at :data:`MEDIA_GROUP_LIMIT`, so larger sets become successive
        albums; a lone image cannot form a media group and is sent as a plain
        photo. This shares the rate-limit gate with alert delivery — one bot
        token means one global limit — so a burst here paces alongside alerts
        instead of racing them. Per-chunk failures are logged and swallowed so a
        partial album still reaches the user rather than aborting the rest.
        """
        for chunk in _chunked(list(items), MEDIA_GROUP_LIMIT):
            try:
                if len(chunk) == 1:
                    image_bytes, caption = chunk[0]
                    self._send_photo_to_chat(chat_id, image_bytes, caption)
                else:
                    self._send_media_group(chat_id, chunk)
            except requests.RequestException as exc:
                LOGGER.warning("Album send to %s failed: %s", chat_id, exc)

    # -- delivery primitives ----------------------------------------------

    def _broadcast(self, user_ids: list[str], send) -> None:
        # Fan out in parallel and wait for all to finish, so the caller can
        # safely delete the snapshot afterwards. map() blocks until every send
        # returns; per-recipient network failures are swallowed so one bad chat
        # never blocks the rest or tears down the camera loop upstream.
        def safe(user_id: str) -> None:
            try:
                send(user_id)
            except requests.RequestException as exc:
                LOGGER.warning("Failed to send alert to %s: %s", user_id, exc)

        list(self._pool.map(safe, user_ids))

    def _send_message(self, user_id: str, text: str) -> None:
        response = self._post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": user_id, "text": text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _send_photo_upload(self, user_id: str, caption: str, image_bytes: bytes) -> str | None:
        response = self._post(
            f"{self.base_url}/sendPhoto",
            data={"chat_id": user_id, "caption": caption},
            files={"photo": ("snapshot.jpg", image_bytes, "image/jpeg")},
            timeout=self.photo_timeout_seconds,
        )
        response.raise_for_status()
        return self._extract_file_id(response)

    def _send_media_group(
        self, chat_id: int | str, chunk: list[tuple[bytes, str | None]]
    ) -> None:
        # Telegram media groups attach each image as a multipart file and
        # reference it from the JSON ``media`` array via ``attach://<key>``.
        media: list[dict[str, str]] = []
        files: dict[str, tuple[str, bytes, str]] = {}
        for index, (image_bytes, caption) in enumerate(chunk):
            key = f"photo{index}"
            entry = {"type": "photo", "media": f"attach://{key}"}
            if caption:
                entry["caption"] = caption
            media.append(entry)
            files[key] = (f"{key}.jpg", image_bytes, "image/jpeg")
        response = self._post(
            f"{self.base_url}/sendMediaGroup",
            data={"chat_id": chat_id, "media": json.dumps(media)},
            files=files,
            timeout=self.photo_timeout_seconds,
        )
        response.raise_for_status()

    def _send_photo_to_chat(
        self, chat_id: int | str, image_bytes: bytes, caption: str | None
    ) -> None:
        # A single image can't be a media group, so it ships as a plain upload.
        data: dict[str, int | str] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        response = self._post(
            f"{self.base_url}/sendPhoto",
            data=data,
            files={"photo": ("snapshot.jpg", image_bytes, "image/jpeg")},
            timeout=self.photo_timeout_seconds,
        )
        response.raise_for_status()

    def _send_photo_by_file_id(self, user_id: str, caption: str, file_id: str) -> None:
        # No upload: Telegram already holds the bytes, so this is a small form
        # post and uses the regular (short) timeout.
        response = self._post(
            f"{self.base_url}/sendPhoto",
            data={"chat_id": user_id, "caption": caption, "photo": file_id},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    # -- rate limiting -----------------------------------------------------

    def _post(self, url: str, **kwargs):
        """POST through the shared gate, honoring 429 retry_after backoff.

        Returns the response (its status may still be an error for the caller's
        ``raise_for_status``); only 429 is retried here.
        """
        attempt = 0
        while True:
            self._acquire_send_slot()
            response = requests.post(url, **kwargs)
            if response.status_code != 429:
                return response

            retry_after = self._retry_after_seconds(response)
            with self._rate_lock:
                self._blocked_until = max(
                    self._blocked_until, time.monotonic() + retry_after
                )
            attempt += 1
            LOGGER.warning(
                "Telegram rate-limited (429); pausing %.1fs before retry %d/%d",
                retry_after,
                attempt,
                self.max_send_retries,
            )
            if attempt > self.max_send_retries:
                # Out of retries: hand back the 429 so the caller's
                # raise_for_status surfaces it and the alert is dropped.
                return response

    def _acquire_send_slot(self) -> None:
        # Block until the gate is clear (no active 429 pause) and the minimum
        # inter-send spacing has elapsed, then reserve the next slot.
        while True:
            with self._rate_lock:
                now = time.monotonic()
                wait = max(self._blocked_until - now, self._next_send_at - now, 0.0)
                if wait <= 0.0:
                    self._next_send_at = now + self.min_send_interval_seconds
                    return
            time.sleep(wait)

    def _retry_after_seconds(self, response) -> float:
        retry_after = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            parameters = payload.get("parameters")
            if isinstance(parameters, dict):
                retry_after = parameters.get("retry_after")
        if retry_after is None:
            header = response.headers.get("Retry-After") if response.headers else None
            if header is not None:
                try:
                    retry_after = float(header)
                except (TypeError, ValueError):
                    retry_after = None
        if retry_after is None:
            retry_after = RETRY_AFTER_FALLBACK_SECONDS
        return float(retry_after) + RETRY_AFTER_BUFFER_SECONDS

    @staticmethod
    def _extract_file_id(response) -> str | None:
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        result = payload.get("result")
        photos = result.get("photo") if isinstance(result, dict) else None
        if not photos:
            return None
        # PhotoSize entries run small -> large; the largest carries the best
        # quality and works as a handle for any chat of this bot.
        best = max(photos, key=lambda size: size.get("file_size", 0))
        return best.get("file_id")
