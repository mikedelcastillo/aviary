"""Telegram alert delivery."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import requests

from aviary_server.detector import Detection


LOGGER = logging.getLogger("aviary_server.telegram")


def parse_command(text: str) -> str | None:
    """Return the bare command from a message text, or ``None``.

    Strips any ``@botname`` suffix and lowercases, so ``/userinfo@AviaryBot``
    and ``/UserInfo`` both yield ``/userinfo``.
    """
    if not text:
        return None
    parts = text.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None
    return parts[0].split("@", 1)[0].lower()


def run_userinfo_bot(
    bot_token: str,
    stop_event: Event | None = None,
    poll_timeout_seconds: int = 30,
) -> None:
    """Long-poll Telegram and reply to ``/userinfo`` with the sender's user ID.

    This is the bootstrap "user_id mode": it does nothing except answer
    ``/userinfo`` so a new operator can discover the ID to put in
    ``TELEGRAM_USER_IDS``. No detection, alerts, or other commands are handled.
    """
    base_url = f"https://api.telegram.org/bot{bot_token}"
    offset: int | None = None
    LOGGER.info("Started in user_id mode; send /userinfo to the bot to get your ID")

    while stop_event is None or not stop_event.is_set():
        try:
            params: dict[str, int] = {"timeout": poll_timeout_seconds}
            if offset is not None:
                params["offset"] = offset
            response = requests.get(
                f"{base_url}/getUpdates",
                params=params,
                timeout=poll_timeout_seconds + 10,
            )
            response.raise_for_status()
            updates = response.json().get("result", [])
        except requests.RequestException as exc:
            LOGGER.warning("getUpdates failed: %s; retrying", exc)
            if stop_event is not None:
                stop_event.wait(5)
            else:
                time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message") or update.get("edited_message")
            if not message:
                continue
            if parse_command(message.get("text", "")) != "/userinfo":
                continue

            user_id = (message.get("from") or {}).get("id")
            chat_id = (message.get("chat") or {}).get("id")
            if chat_id is None or user_id is None:
                continue

            text = f"Your Telegram user ID is: {user_id}\nAdd it to TELEGRAM_USER_IDS to enable alerts."
            try:
                requests.post(
                    f"{base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=15,
                ).raise_for_status()
                LOGGER.info("Replied /userinfo for user %s", user_id)
            except requests.RequestException as exc:
                LOGGER.warning("Failed to reply to /userinfo: %s", exc)


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        user_ids: list[str],
        timeout_seconds: int = 15,
        photo_timeout_seconds: int = 60,
    ) -> None:
        self.bot_token = bot_token
        self.user_ids = user_ids
        self.timeout_seconds = timeout_seconds
        # Photo uploads are far heavier than text and slow uplinks need headroom.
        self.photo_timeout_seconds = photo_timeout_seconds
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        # Deliver to all recipients concurrently so 3-4 recipients cost one
        # upload's worth of latency, not the sum. Sized to the recipient count.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, len(user_ids)),
            thread_name_prefix="telegram-send",
        )

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

        # Fan out to all recipients in parallel and wait for them to finish, so
        # the caller can safely delete the snapshot afterwards. map() blocks
        # until every send returns; failures are swallowed inside _deliver.
        list(
            self._pool.map(
                lambda user_id: self._deliver(user_id, text, snapshot_path if has_photo else None),
                self.user_ids,
            )
        )

    def _deliver(self, user_id: str, text: str, snapshot_path: Path | None) -> None:
        # Network failures here must never propagate: this runs inside the camera
        # capture loop, and an unhandled error would tear down and reconnect the
        # RTSP stream. A dropped alert is logged and swallowed.
        try:
            if snapshot_path is not None:
                self._send_photo(user_id, text, snapshot_path)
            else:
                self._send_message(user_id, text)
        except requests.RequestException as exc:
            LOGGER.warning("Failed to send alert to %s: %s", user_id, exc)

    def _send_message(self, user_id: str, text: str) -> None:
        response = requests.post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": user_id, "text": text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _send_photo(self, user_id: str, caption: str, snapshot_path: Path) -> None:
        with snapshot_path.open("rb") as image_file:
            response = requests.post(
                f"{self.base_url}/sendPhoto",
                data={"chat_id": user_id, "caption": caption},
                files={"photo": image_file},
                timeout=self.photo_timeout_seconds,
            )
        response.raise_for_status()
