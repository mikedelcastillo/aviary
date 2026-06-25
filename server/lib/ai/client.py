"""Minimal HTTP client for a local Ollama server.

Wraps the two endpoints we need — ``/api/chat`` (language: intent + chat) and
``/api/generate`` (vision: image captioning) — over ``requests``, with no
streaming and no extra dependency. The server points at whatever Ollama the
machine already runs (``OLLAMA_BASE_URL``); this module never starts or installs
it.

Calls raise on transport/HTTP error so callers can decide how to degrade; a
:meth:`is_available` probe is offered for a one-shot startup check.
"""

from __future__ import annotations

import logging
from typing import Any

import requests


LOGGER = logging.getLogger("lib.ai.client")


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def is_available(self) -> bool:
        """Best-effort reachability probe (lists installed models)."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        fmt: Any | None = None,
        think: bool | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """One non-streaming /api/chat turn; returns the assistant message text.

        ``fmt`` is Ollama's ``format`` (``"json"`` or a JSON schema) for
        structured output. ``think=False`` disables a thinking model's
        deliberation so intent parsing stays fast.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if fmt is not None:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        response = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout_seconds or self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("message") or {}).get("content", "") or ""

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        images: list[str] | None = None,
        fmt: Any | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """One non-streaming /api/generate call; returns the response text.

        ``images`` are base64-encoded JPEG/PNG strings for a vision model.
        """
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if images:
            payload["images"] = images
        if fmt is not None:
            payload["format"] = fmt
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        response = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=timeout_seconds or self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("response", "") or ""
