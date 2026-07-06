"""Live availability of the Ollama/Olla backend, so callers can fail FAST.

Without this, every feature discovers an outage on its own: each call eats the
full connect-retry ladder (3 attempts + backoff, up to the 120s timeout) before
its caller degrades, and nothing knows when the cluster comes back. This monitor
polls cheaply in the background and answers two questions instantly:

  * :attr:`available` — is at least one worker able to serve requests right now?
  * :meth:`has_model` — is a given model actually served by the cluster?

The :class:`~lib.ai.client.OllamaClient` gates its calls on ``available`` (an
outage raises :class:`~lib.ai.client.OllamaUnavailableError` immediately instead
of hanging), and feeds real request outcomes back via :meth:`note_success` /
:meth:`note_failure` so state flips as soon as evidence arrives, not on the next
poll. Recovery listeners (:meth:`on_change`) let the backfill worker wake the
moment the cluster returns.

Probes: ``GET {base}/api/tags`` gives reachability + the served model list for a
plain Ollama AND for Olla. Behind Olla, ``/internal/status`` is ALSO probed —
Olla itself stays up (and its unified model registry can hold stale entries for
24h) while every worker behind it is down, so the per-endpoint health there is
the authoritative signal.

Availability is also MODEL-scoped. A cluster can be "up" (some worker healthy)
while nobody serves one particular model — e.g. the only box with the vision
model is off overnight while a small worker keeps chat alive. Olla offers no
complete model→healthy-endpoint map (its registry is a stale union; its
``/internal/status/models`` lists only recent entries), so the truth comes from
real requests: :meth:`note_model_missing` (Olla's 404 "No ollama endpoints
available" for the model, or a 503) gates that one model for
:data:`MODEL_MISSING_TTL_SECONDS`, after which the next attempt is let through
to re-test. :meth:`has_model` folds this evidence in, so gated callers degrade
per-model instead of flapping the whole backend down and up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

import requests


LOGGER = logging.getLogger("lib.ai.health")

# Poll gently while healthy; much faster while down so recovery (and the queued
# message replay that follows) is noticed within seconds.
UP_POLL_SECONDS = 30.0
DOWN_POLL_SECONDS = 8.0
PROBE_TIMEOUT_SECONDS = 5.0

# Olla endpoint states that can (or will shortly) serve a request. A single-GPU
# worker flaps "offline" mid-inference (it can't answer the health check while
# generating), so only count it out when Olla says so — but "busy"/"warming"
# style states still count as capacity.
_SERVING_STATES = {"healthy", "busy", "warming", "degraded"}

# How long a model stays gated after real request evidence said nobody serves
# it. Long enough that a dead-overnight worker doesn't get hammered with a
# doomed call per memory beat; short enough that ONE half-open re-test per
# window notices the worker returning even when overall availability never
# flipped (so no on_change fires to clear the marker).
MODEL_MISSING_TTL_SECONDS = 180.0


def derive_status_url(base_url: str) -> str | None:
    """Olla's ``/internal/status`` URL for an Olla base URL, else None.

    An Olla base URL carries the proxy path (e.g. ``…:40114/olla/ollama``); a
    plain Ollama URL has no path, and has no status endpoint to ask.
    """
    parts = urlsplit(base_url)
    if "/olla" in (parts.path or ""):
        return f"{parts.scheme}://{parts.netloc}/internal/status"
    return None


class OllamaHealth:
    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        status_url: str | None = None,
        up_poll_seconds: float = UP_POLL_SECONDS,
        down_poll_seconds: float = DOWN_POLL_SECONDS,
        probe_timeout: float = PROBE_TIMEOUT_SECONDS,
        model_missing_ttl: float = MODEL_MISSING_TTL_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Own session (not the client's): probes must never contend with a busy
        # generation for a pooled connection.
        self._session = session or requests.Session()
        self._status_url = status_url if status_url is not None else derive_status_url(base_url)
        self._up_poll = up_poll_seconds
        self._down_poll = down_poll_seconds
        self._probe_timeout = probe_timeout
        self._lock = threading.Lock()
        # Optimistic until the first probe: gating every call "down" during the
        # first seconds of boot would queue messages needlessly.
        self._available = True
        self._models: frozenset[str] = frozenset()
        self._healthy_endpoints: int | None = None
        self._listeners: list[Callable[[bool], None]] = []
        # Poked by note_failure()/note_success() so the poll loop re-probes (and
        # re-paces) immediately instead of sleeping out the healthy interval.
        self._poke = threading.Event()
        # Bumped on every note_failure: a probe that started BEFORE a real
        # request failed must not commit its stale UP verdict over that fresher
        # evidence (probe network I/O takes seconds; failures are instant).
        self._failure_evidence = 0
        # Model → monotonic time it was last reported unserveable by a real
        # request. The registry can't be trusted for this (Olla's is a stale
        # union), so request evidence gates the model until the TTL lets one
        # half-open re-test through.
        self._model_missing_ttl = model_missing_ttl
        self._model_missing: dict[str, float] = {}

    # -- state ---------------------------------------------------------------

    @property
    def available(self) -> bool:
        with self._lock:
            return self._available

    @property
    def models(self) -> frozenset[str]:
        with self._lock:
            return self._models

    @property
    def healthy_endpoints(self) -> int | None:
        """How many Olla workers can serve right now (None when not behind Olla)."""
        with self._lock:
            return self._healthy_endpoints

    def has_model(self, name: str) -> bool:
        """Whether the cluster serves ``name`` — optimistic when the list is unknown.

        A bare name ("gemma3") matches any tag of it; a tagged name must match
        exactly. Callers use this to skip a model that ISN'T served (e.g. the
        recall model when only the small worker is up) without a doomed request.

        Request evidence beats the registry: a model recently reported by
        :meth:`note_model_missing` is NOT served no matter what /api/tags says —
        behind Olla that list is a union registry that keeps a dead worker's
        models for up to 24h. Once the marker's TTL lapses it stops gating, so
        the next real call re-tests the model. Reading NEVER consumes the
        marker — a mere availability check (e.g. the backfill drain's guard)
        must not swallow the half-open slot; the re-test's own outcome refreshes
        the marker (:meth:`note_model_missing`) or clears it
        (:meth:`note_model_served`).
        """
        name = (name or "").strip()
        if not name:
            return False
        with self._lock:
            marked = self._model_missing.get(name)
            if marked is not None and time.monotonic() - marked < self._model_missing_ttl:
                return False
        models = self.models
        if not models:
            return True
        if name in models:
            return True
        if ":" not in name:
            return any(m.split(":", 1)[0] == name for m in models)
        return False

    def on_change(self, callback: Callable[[bool], None]) -> None:
        """Register a listener called with the new availability on each flip."""
        with self._lock:
            self._listeners.append(callback)

    # -- evidence from real requests ------------------------------------------

    def note_success(self) -> None:
        """A real request succeeded — the backend is definitively up."""
        self._set_available(True)

    def note_failure(self) -> None:
        """A real request failed at the transport level — treat as down NOW.

        Flipping immediately (rather than waiting for the next poll) means the
        very next caller queues instantly instead of eating its own retry
        ladder. The poll loop is poked to confirm and re-pace.
        """
        with self._lock:
            self._failure_evidence += 1
        self._set_available(False)

    def note_model_missing(self, model: str) -> None:
        """A real request said nobody serves ``model`` (Olla's 404, or a 503).

        Model-scoped evidence: the rest of the cluster may be fine — chat keeps
        working on a small worker while the only vision box is off overnight —
        so this gates ONLY this model's calls (via :meth:`has_model`), instead
        of flapping the whole backend down and up. Cleared by
        :meth:`note_model_served`, by TTL lapse, or by a recovery flip.
        """
        model = (model or "").strip()
        if not model:
            return
        with self._lock:
            known = model in self._model_missing
            self._model_missing[model] = time.monotonic()
        if not known:
            LOGGER.warning(
                "No healthy backend serves %s — gating its calls (re-test in %.0fs)",
                model, self._model_missing_ttl,
            )

    def note_model_served(self, model: str) -> None:
        """A real request for ``model`` succeeded — drop its missing marker."""
        with self._lock:
            marked = self._model_missing.pop((model or "").strip(), None)
        if marked is not None:
            LOGGER.warning("Model %s is served again — its calls resume", model)

    def poke(self) -> None:
        """Ask the poll loop to re-probe now (e.g. a 503 hints at a wider outage)."""
        self._poke.set()

    # -- probing ---------------------------------------------------------------

    def probe(self) -> bool:
        """One synchronous check; updates state and returns availability."""
        with self._lock:
            evidence_at_start = self._failure_evidence
        up = False
        models: frozenset[str] | None = None
        try:
            response = self._session.get(f"{self._base_url}/api/tags", timeout=self._probe_timeout)
            if response.ok:
                up = True
                try:
                    models = frozenset(
                        str(m.get("name", "")).strip()
                        for m in (response.json().get("models") or [])
                        if str(m.get("name", "")).strip()
                    )
                except (ValueError, AttributeError):
                    models = None
        except requests.RequestException:
            up = False

        healthy: int | None = None
        if self._status_url:
            try:
                response = self._session.get(self._status_url, timeout=self._probe_timeout)
                if response.ok:
                    endpoints = response.json().get("endpoints") or []
                    healthy = sum(
                        1
                        for e in endpoints
                        if str(e.get("status", "")).strip().lower() in _SERVING_STATES
                    )
            except (requests.RequestException, ValueError, AttributeError):
                # Olla's status being unreadable is not evidence the workers are
                # down — fall back to the /api/tags verdict alone.
                healthy = None
        if healthy is not None:
            # Olla answers /api/tags from its (up to 24h stale) unified registry
            # even with every worker offline — endpoint health is authoritative.
            up = up and healthy > 0

        with self._lock:
            if models is not None:
                self._models = models
            self._healthy_endpoints = healthy
            # An UP verdict is stale if a real request failed while this probe's
            # network I/O was in flight — keep the gate closed; the (poked) loop
            # re-probes immediately. DOWN verdicts always commit.
            if up and self._failure_evidence != evidence_at_start:
                return self._available
        self._set_available(up)
        return up

    def _set_available(self, value: bool) -> None:
        with self._lock:
            changed = self._available != value
            self._available = value
            if changed and value:
                # Recovery invalidates per-model evidence gathered during the
                # outage: the workers that just returned may well be the ones
                # serving those models. Let real requests re-test immediately.
                self._model_missing.clear()
            listeners = list(self._listeners) if changed else []
        if not changed:
            return
        self._poke.set()
        LOGGER.warning("Ollama backend %s", "recovered — back ONLINE" if value else "went OFFLINE")
        for callback in listeners:
            try:
                callback(value)
            except Exception:
                LOGGER.exception("Ollama health listener failed")

    # -- background loop --------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        threading.Thread(
            target=self._run, args=(stop_event,), name="ollama-health", daemon=True
        ).start()

    def _run(self, stop_event: threading.Event) -> None:
        LOGGER.info(
            "Ollama health monitor on (%s%s)",
            self._base_url,
            f", status {self._status_url}" if self._status_url else "",
        )
        while not stop_event.is_set():
            self.probe()
            interval = self._up_poll if self.available else self._down_poll
            self._poke.wait(interval)
            self._poke.clear()
