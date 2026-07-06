from __future__ import annotations

import requests

from lib.ai.health import OllamaHealth, derive_status_url


class FakeResponse:
    def __init__(self, payload: dict, *, ok: bool = True, status: int = 200) -> None:
        self._payload = payload
        self.ok = ok
        self.status_code = status

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Routes GETs by URL; a value that is an Exception instance is raised."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes
        self.gets: list[tuple] = []

    def get(self, url, timeout):
        self.gets.append((url, timeout))
        result = self.routes[url]
        if isinstance(result, Exception):
            raise result
        return result


def test_derive_status_url_for_olla_base() -> None:
    # An Olla base URL carries the proxy path; the status endpoint lives at the
    # host root, not under the proxy path.
    assert (
        derive_status_url("http://host:40114/olla/ollama")
        == "http://host:40114/internal/status"
    )


def test_derive_status_url_plain_ollama_is_none() -> None:
    # Plain Ollama has no /internal/status to ask — probing one would 404.
    assert derive_status_url("http://host:11434") is None


def test_probe_tags_ok_no_status_url_parses_models() -> None:
    session = FakeSession(
        {
            "http://host:11434/api/tags": FakeResponse(
                {"models": [{"name": "gemma3:4b"}, {"name": "qwen2.5vl:7b"}]}
            )
        }
    )
    health = OllamaHealth("http://host:11434", session=session)
    assert health.probe() is True
    assert health.available is True
    assert health.models == frozenset({"gemma3:4b", "qwen2.5vl:7b"})
    # Not behind Olla: no per-endpoint health to report.
    assert health.healthy_endpoints is None


def test_probe_connection_error_marks_down() -> None:
    session = FakeSession(
        {"http://host:11434/api/tags": requests.ConnectionError("refused")}
    )
    health = OllamaHealth("http://host:11434", session=session)
    assert health.probe() is False
    assert health.available is False


def test_probe_stale_olla_registry_does_not_mask_dead_cluster() -> None:
    # THE KEY CASE: Olla answers /api/tags from its unified registry (up to 24h
    # stale) even when every worker is offline. Endpoint health is authoritative:
    # zero serving endpoints must mean unavailable despite a happy /api/tags.
    base = "http://host:40114/olla/ollama"
    session = FakeSession(
        {
            f"{base}/api/tags": FakeResponse({"models": [{"name": "gemma3:4b"}]}),
            "http://host:40114/internal/status": FakeResponse(
                {"endpoints": [{"status": "offline"}, {"status": "offline"}]}
            ),
        }
    )
    health = OllamaHealth(base, session=session)
    assert health.probe() is False
    assert health.available is False
    assert health.healthy_endpoints == 0


def test_probe_unreadable_status_falls_back_to_tags_verdict() -> None:
    # Olla's status endpoint being broken is not evidence the workers are down;
    # the /api/tags verdict alone must stand.
    base = "http://host:40114/olla/ollama"
    session = FakeSession(
        {
            f"{base}/api/tags": FakeResponse({"models": [{"name": "gemma3:4b"}]}),
            "http://host:40114/internal/status": requests.ConnectionError("boom"),
        }
    )
    health = OllamaHealth(base, session=session)
    assert health.probe() is True
    assert health.available is True
    assert health.healthy_endpoints is None


def test_has_model_matching_rules() -> None:
    session = FakeSession(
        {
            "http://host:11434/api/tags": FakeResponse(
                {"models": [{"name": "gemma3:12b"}, {"name": "qwen2.5vl:7b"}]}
            )
        }
    )
    health = OllamaHealth("http://host:11434", session=session)
    health.probe()
    # Exact tagged match.
    assert health.has_model("qwen2.5vl:7b") is True
    # A bare name matches any tag of it.
    assert health.has_model("gemma3") is True
    # A tagged name must match exactly — a different tag is not served.
    assert health.has_model("gemma3:4b") is False


def test_has_model_optimistic_when_model_list_unknown() -> None:
    # Before the first successful probe the served-model list is unknown; assume
    # the model exists rather than wrongly skipping a servable request.
    health = OllamaHealth("http://host:11434", session=FakeSession({}))
    assert health.has_model("anything:1b") is True


def test_note_model_missing_gates_has_model_despite_registry() -> None:
    # THE OVERNIGHT CASE: the registry (Olla's stale union) still lists the
    # vision model after its only worker went off; request evidence must win so
    # vision calls gate instantly instead of 502-storming per memory beat.
    session = FakeSession(
        {
            "http://host:11434/api/tags": FakeResponse(
                {"models": [{"name": "gemma3:4b"}, {"name": "qwen2.5vl:7b"}]}
            )
        }
    )
    health = OllamaHealth("http://host:11434", session=session)
    health.probe()
    health.note_model_missing("qwen2.5vl:7b")
    assert health.has_model("qwen2.5vl:7b") is False
    # Model-scoped: the rest of the cluster is untouched.
    assert health.has_model("gemma3:4b") is True
    assert health.available is True


def test_note_model_served_clears_the_marker() -> None:
    health = OllamaHealth("http://host:11434", session=FakeSession({}))
    health.note_model_missing("qwen2.5vl:7b")
    assert health.has_model("qwen2.5vl:7b") is False
    health.note_model_served("qwen2.5vl:7b")
    assert health.has_model("qwen2.5vl:7b") is True


def test_model_missing_marker_expires_for_a_half_open_retest() -> None:
    # After the TTL real calls must get through to re-test the model — overall
    # availability may never flip (a small worker stayed healthy), so no
    # recovery listener will ever clear the marker for us.
    health = OllamaHealth(
        "http://host:11434", session=FakeSession({}), model_missing_ttl=0.0
    )
    health.note_model_missing("qwen2.5vl:7b")
    assert health.has_model("qwen2.5vl:7b") is True  # TTL already lapsed
    # Reading must NOT consume the marker: a mere guard check (the backfill
    # drain asks has_model before replaying queued messages) would otherwise
    # swallow the half-open slot, and the doomed replay that follows would burn
    # one of the message's give-up attempts. Only the re-test's own outcome
    # moves the marker: note_model_missing refreshes it, note_model_served
    # clears it.
    assert "qwen2.5vl:7b" in health._model_missing
    assert health.has_model("qwen2.5vl:7b") is True  # idempotent read


def test_model_missing_beats_optimistic_unknown_registry() -> None:
    # Even with no model list yet (optimistic mode), fresh request evidence
    # that nobody serves the model must gate it.
    health = OllamaHealth("http://host:11434", session=FakeSession({}))
    assert health.has_model("qwen2.5vl:7b") is True  # optimistic
    health.note_model_missing("qwen2.5vl:7b")
    assert health.has_model("qwen2.5vl:7b") is False


def test_recovery_flip_clears_all_model_markers() -> None:
    # A full-outage recovery usually IS the missing models' worker returning;
    # stale per-model markers must not keep gating for their whole TTL.
    health = OllamaHealth("http://host:11434", session=FakeSession({}))
    health.note_model_missing("qwen2.5vl:7b")
    health.note_model_missing("gemma3:12b")
    health.note_failure()  # flips down
    health.note_success()  # flips up — recovery
    assert health.has_model("qwen2.5vl:7b") is True
    assert health.has_model("gemma3:12b") is True


def test_poke_wakes_the_poll_loop() -> None:
    health = OllamaHealth("http://host:11434", session=FakeSession({}))
    assert not health._poke.is_set()
    health.poke()
    assert health._poke.is_set()


def test_note_failure_and_success_fire_listener_only_on_flips() -> None:
    health = OllamaHealth("http://host:11434", session=FakeSession({}))
    calls: list[bool] = []
    health.on_change(calls.append)

    # Starts optimistic (True); a transport failure flips down and fires once.
    health.note_failure()
    assert health.available is False
    assert calls == [False]

    # Same state again: no flip, no listener call.
    health.note_failure()
    assert calls == [False]

    # Recovery evidence flips back up and fires once with True.
    health.note_success()
    assert health.available is True
    assert calls == [False, True]

    # Redundant success: still no extra call.
    health.note_success()
    assert calls == [False, True]


def test_poll_loop_flips_down_then_recovers_and_fires_listener() -> None:
    # The background loop is the ONLY recovery path once the gate closes: it
    # must notice a dead backend, then flip back up (firing on_change) when the
    # backend returns — without any real-request note_success().
    import threading
    import time as _time

    class MutableSession:
        def __init__(self) -> None:
            self.up = True

        def get(self, url, timeout):
            if not self.up:
                raise requests.ConnectionError("refused")
            return FakeResponse({"models": [{"name": "gemma3:4b"}]})

    session = MutableSession()
    health = OllamaHealth(
        "http://host:11434", session=session,
        up_poll_seconds=0.02, down_poll_seconds=0.02,
    )
    flips: list[bool] = []
    flipped = threading.Event()

    def listener(up: bool) -> None:
        flips.append(up)
        flipped.set()

    health.on_change(listener)
    stop = threading.Event()
    health.start(stop)
    try:
        session.up = False
        assert flipped.wait(2.0), "never flipped down"
        assert health.available is False
        flipped.clear()
        session.up = True
        assert flipped.wait(2.0), "never recovered"
        assert health.available is True
        assert flips[:2] == [False, True]
    finally:
        stop.set()


def test_probe_up_verdict_is_discarded_when_failure_evidence_arrives_mid_probe() -> None:
    # A probe's network I/O takes time; a real request can fail while it's in
    # flight. The probe's stale UP verdict must not reopen the gate over that
    # fresher DOWN evidence.
    health_ref: list = []

    class RacingSession:
        def get(self, url, timeout):
            # The outage is discovered by a real request while this probe runs.
            health_ref[0].note_failure()
            return FakeResponse({"models": [{"name": "gemma3:4b"}]})

    health = OllamaHealth("http://host:11434", session=RacingSession())
    health_ref.append(health)
    assert health.probe() is False  # the stale UP verdict was NOT committed
    assert health.available is False
    # A later clean probe (no mid-flight failures) recovers normally.
    health._session = FakeSession(
        {"http://host:11434/api/tags": FakeResponse({"models": [{"name": "gemma3:4b"}]})}
    )
    assert health.probe() is True
    assert health.available is True
