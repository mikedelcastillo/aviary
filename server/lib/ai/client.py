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
import threading
import time
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
        vision_concurrency: int = 1,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        # Transient 5xx / connection errors are retried — important behind an Olla
        # load balancer, where a single backend can be briefly busy (502) while
        # another is healthy; a retry routes around it.
        self._max_retries = max(0, max_retries)
        self._retry_backoff = retry_backoff_seconds
        # Vision calls are the expensive ones, and many fire at once (naming all
        # cameras, captioning every memory frame, a /find description). A bounded
        # semaphore turns generate() into a queue so the GPU isn't swamped — the
        # extra callers block here and run as slots free, instead of all at once.
        self._vision_slots = threading.BoundedSemaphore(max(1, vision_concurrency))

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
        data = self._post_json("/api/chat", payload, timeout_seconds or self.timeout_seconds)
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
        # Queue behind the vision semaphore so concurrent callers don't pile onto
        # the GPU at once; the timeout only starts once we hold a slot.
        with self._vision_slots:
            data = self._post_json("/api/generate", payload, timeout_seconds or self.timeout_seconds)
            return data.get("response", "") or ""

    def _post_json(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """POST JSON and return the parsed body, retrying transient failures.

        A 5xx (e.g. an Olla backend returning 502 while busy) or a connection
        error is retried up to ``max_retries`` times with a short backoff; the
        last error is raised if every attempt fails.
        """
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                # A 4xx is usually a client error (bad request, auth, not-found)
                # that a retry can't fix — fail fast rather than burning the
                # backoff. The exceptions are 408 (timeout) and 429 (rate limit),
                # which ARE transient (a busy Olla backend) and worth retrying,
                # like 5xx and connection/timeout errors.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status not in (408, 429):
                    raise
                if attempt < self._max_retries:
                    LOGGER.warning(
                        "Ollama %s failed (%s); retry %d/%d",
                        path, exc, attempt + 1, self._max_retries,
                    )
                    time.sleep(self._retry_backoff)
        raise last_exc  # type: ignore[misc]
