"""Minimal HTTP client for a local Ollama server.

Wraps the two endpoints we need — ``/api/chat`` (language: intent + chat) and
``/api/generate`` (vision: image captioning) — over ``requests``, with no
streaming and no extra dependency. The server points at whatever Ollama the
machine already runs (``OLLAMA_BASE_URL``); this module never starts or installs
it.

Calls raise on transport/HTTP error so callers can decide how to degrade; a
:meth:`is_available` probe is offered for a one-shot startup check. When built
with a :class:`~lib.ai.health.OllamaHealth`, calls during a known outage raise
:class:`OllamaUnavailableError` IMMEDIATELY — no connect attempts, no retry
ladder — so callers can queue the work for backfill instead of hanging.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests


LOGGER = logging.getLogger("lib.ai.client")

# How long a vision caller may wait for a semaphore slot before giving up. The
# wait used to be unbounded, so a wedged generation could strand every later
# caller forever; anything past this is a pathological queue, not a busy one.
VISION_SLOT_WAIT_SECONDS = 300.0


class OllamaUnavailableError(requests.exceptions.ConnectionError):
    """The backend is known-down (health-gated); raised without any request.

    Also raised when the backend is up but NOBODY serves the requested model
    (the only worker with the vision model is off overnight) — same treatment:
    the work is queued/backfilled rather than retried into a doomed call.

    Subclasses ``ConnectionError`` so every existing ``RequestException``
    handler degrades exactly as it would for a real refused connection — just
    instantly.
    """


class OllamaBusyError(requests.exceptions.ConnectionError):
    """No vision slot freed up within the wait budget; the call never started."""


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
        health=None,
        keep_alive: str | None = None,
        usage=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        # Transient 5xx / connection errors are retried — important behind an Olla
        # load balancer, where a single backend can be briefly busy (502) while
        # another is healthy; a retry routes around it.
        self._max_retries = max(0, max_retries)
        self._retry_backoff = retry_backoff_seconds
        # Optional lib.ai.health.OllamaHealth: gates calls during a known outage
        # and receives success/failure evidence from every real request.
        self._health = health
        # Optional lib.ai.usage.LlmUsageRecorder: per-model token/latency
        # accounting from each response's own metering fields (for /tokens).
        self._usage = usage
        # Ollama keep_alive ("30m", "-1" = forever): how long a worker keeps the
        # model loaded after a request. The requests here are sporadic (a chat
        # turn, a 5-min memory beat), so without this the default 5m unload means
        # nearly every call pays a cold model load first.
        self._keep_alive = keep_alive
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
        num_predict: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """One non-streaming /api/chat turn; returns the assistant message text.

        ``fmt`` is Ollama's ``format`` (``"json"`` or a JSON schema) for
        structured output. ``think=False`` disables a thinking model's
        deliberation so intent parsing stays fast. ``num_predict`` caps the
        generated tokens — a reasoning model (qwen3) with no ceiling can decode
        thousands of tokens per call and, on a CPU-spilled backend, peg the CPU
        for minutes; capping it bounds the worst case.
        """
        self._gate(model)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if fmt is not None:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        if num_predict is not None:
            payload.setdefault("options", {})["num_predict"] = num_predict
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
        num_predict: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """One non-streaming /api/generate call; returns the response text.

        ``images`` are base64-encoded JPEG/PNG strings for a vision model.
        ``num_predict`` caps the generated tokens (bounds a runaway caption).
        """
        self._gate(model)
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if images:
            payload["images"] = images
        if fmt is not None:
            payload["format"] = fmt
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        if num_predict is not None:
            payload.setdefault("options", {})["num_predict"] = num_predict
        # Queue behind the vision semaphore so concurrent callers don't pile onto
        # the GPU at once; the timeout only starts once we hold a slot. The wait
        # itself is bounded — one wedged generation must not strand every later
        # vision caller forever.
        if not self._vision_slots.acquire(timeout=VISION_SLOT_WAIT_SECONDS):
            raise OllamaBusyError(
                f"no vision slot freed within {VISION_SLOT_WAIT_SECONDS:.0f}s"
            )
        try:
            data = self._post_json("/api/generate", payload, timeout_seconds or self.timeout_seconds)
            return data.get("response", "") or ""
        finally:
            self._vision_slots.release()

    def _gate(self, model: str | None = None) -> None:
        """Fail a call instantly during a KNOWN outage (no connect, no retries).

        The health monitor's verdict may be a few seconds stale; a just-recovered
        backend is picked up by its fast down-poll, and the queued work is
        replayed by the backfill worker moments later.

        The gate is also MODEL-scoped: a cluster counts as "up" with any worker
        healthy, but that worker may not serve this call's model (only the
        vision box has the VLM, and it's off). Those calls fail instantly too —
        as the same ConnectionError subclass — so they queue/backfill instead of
        eating a doomed request per memory beat all night.
        """
        if self._health is None:
            return
        if not self._health.available:
            raise OllamaUnavailableError(
                f"Ollama backend at {self.base_url} is down (health-gated)"
            )
        if model and not self._health.has_model(model):
            raise OllamaUnavailableError(
                f"no healthy Ollama backend serves {model} right now (health-gated)"
            )

    def _post_json(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """POST JSON and return the parsed body, retrying transient failures.

        A 5xx (e.g. an Olla backend returning 502 while busy) or a connection
        error is retried up to ``max_retries`` times with a short backoff; the
        last error is raised if every attempt fails. Outcomes are reported to the
        health monitor (so an outage discovered here gates the NEXT caller
        immediately) and to the usage recorder (per-model token accounting).
        """
        try:
            data = self._post_json_attempts(path, payload, timeout)
        except requests.RequestException:
            # Metering must never change the primary path's exception type —
            # guard the recorder so the ORIGINAL transport error propagates.
            if self._usage is not None:
                try:
                    self._usage.record_failure(str(payload.get("model", "")))
                except Exception:
                    LOGGER.exception("Recording LLM failure failed")
            raise
        if self._usage is not None:
            try:
                self._usage.record(str(payload.get("model", "")), data)
            except Exception:
                LOGGER.exception("Recording LLM usage failed")
        return data

    def _post_json_attempts(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        model = str(payload.get("model", "") or "")
        last_exc: Exception | None = None
        # Split (connect, read) timeout: a single value would let a black-holed
        # host hang the CONNECT for the whole generous read budget. Connecting is
        # local/LAN — 10s is already generous.
        timeouts = (min(10.0, timeout), timeout)
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=timeouts)
                response.raise_for_status()
                data = response.json()
                if self._health is not None:
                    self._health.note_success()
                    if model:
                        self._health.note_model_served(model)
                return data
            except requests.RequestException as exc:
                last_exc = exc
                # Transport-level failures (refused/unreachable/connect-timeout)
                # mean the whole backend is gone — outage evidence. A 404 (Olla:
                # "No ollama endpoints available" for this model) or 503 (no
                # healthy backend for THIS request) is MODEL-scoped: the cluster
                # may still serve everything else, so gate just the model —
                # slamming the global gate here is what flapped the backend
                # down/up all night when only the vision worker was off. Those
                # two re-raise as OllamaUnavailableError (a ConnectionError):
                # every outage handler then treats the failed re-test like the
                # gated calls it follows — queue/pause, never a burned give-up
                # attempt. A 503 also pokes a probe, in case it really is a full
                # outage (the probe's endpoint counts decide that within
                # seconds). Anything ambiguous — ReadTimeout, 502 on one busy
                # worker, 429, a request-specific 500 (one poison photo can
                # reproducibly crash a runner) — is evidence of NOTHING: gating
                # a served model on it would stall every caller for the TTL.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if self._health is not None:
                    if isinstance(
                        exc, (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)
                    ):
                        self._health.note_failure()
                    elif status in (404, 503) and model:
                        self._health.note_model_missing(model)
                        if status == 503:
                            self._health.poke()
                        raise OllamaUnavailableError(
                            f"no Ollama backend serves {model} right now (HTTP {status})"
                        ) from exc
                # A 4xx is usually a client error (bad request, auth, not-found)
                # that a retry can't fix — fail fast rather than burning the
                # backoff. The exceptions are 408 (timeout) and 429 (rate limit),
                # which ARE transient (a busy Olla backend) and worth retrying,
                # like 5xx and connection/timeout errors.
                if status is not None and 400 <= status < 500 and status not in (408, 429):
                    raise
                # A ReadTimeout means the backend accepted the request and is
                # still generating server-side; retrying does NOT cancel that
                # work, it spawns a *second* full generation on top of the first
                # and triples CPU on an already-slow backend. Fail fast instead.
                # (A ConnectTimeout — never reached the backend — still retries.)
                if isinstance(exc, requests.exceptions.ReadTimeout):
                    raise
                if attempt < self._max_retries:
                    LOGGER.warning(
                        "Ollama %s failed (%s); retry %d/%d",
                        path, exc, attempt + 1, self._max_retries,
                    )
                    time.sleep(self._retry_backoff)
        raise last_exc  # type: ignore[misc]
