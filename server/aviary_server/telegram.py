"""Telegram alert delivery."""

from __future__ import annotations

import logging
import time
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
    def __init__(self, bot_token: str, user_ids: list[str], timeout_seconds: int = 15) -> None:
        self.bot_token = bot_token
        self.user_ids = user_ids
        self.timeout_seconds = timeout_seconds
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_detections(
        self,
        camera_name: str,
        detections: list[Detection],
        snapshot_path: Path | None,
    ) -> None:
        if not self.bot_token or not self.user_ids:
            raise ValueError("Telegram bot token and user IDs are required")

        labels = ", ".join(sorted({detection.label for detection in detections}))
        zones = sorted({detection.zone for detection in detections if detection.zone})
        zone_text = f"\nZones: {', '.join(zones)}" if zones else ""
        text = f"Aviary alert\nCamera: {camera_name}\nBirds: {labels}{zone_text}"

        for user_id in self.user_ids:
            if snapshot_path and snapshot_path.exists():
                self._send_photo(user_id, text, snapshot_path)
            else:
                self._send_message(user_id, text)

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
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
