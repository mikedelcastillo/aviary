"""Telegram alert delivery."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from lib.detector import Detection


LOGGER = logging.getLogger("lib.telegram")


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
