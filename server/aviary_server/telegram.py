"""Telegram alert delivery."""

from __future__ import annotations

from pathlib import Path

import requests

from aviary_server.detector import Detection


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
