"""Telegram bootstrap bot for discovering user IDs."""

from __future__ import annotations

import logging
import time
from threading import Event

import requests


LOGGER = logging.getLogger("lib.telegram.userinfo")


def parse_command(text: str) -> str | None:
    """Return the bare command from a message text, or ``None``."""
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
    """Long-poll Telegram and reply to ``/userinfo`` with the sender's user ID."""
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
