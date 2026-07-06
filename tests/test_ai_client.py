from __future__ import annotations

import requests

from lib.ai.client import OllamaClient


class FakeResponse:
    def __init__(self, payload: dict, *, ok: bool = True, status: int = 200) -> None:
        self._payload = payload
        self.ok = ok
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            # Mirror requests: the raised HTTPError carries the response, so the
            # client can inspect the status code to decide whether to retry.
            error = requests.HTTPError(f"status {self.status_code}")
            error.response = self  # type: ignore[attr-defined]
            raise error


class FakeSession:
    def __init__(self, response) -> None:
        self.response = response
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        return self.response

    def get(self, url, timeout):
        self.gets.append((url, timeout))
        return self.response


def test_chat_builds_payload_and_returns_content() -> None:
    session = FakeSession(FakeResponse({"message": {"content": "hello"}}))
    client = OllamaClient("http://ollama:11434/", session=session)

    out = client.chat(
        "qwen3:4b",
        [{"role": "user", "content": "hi"}],
        fmt={"type": "object"},
        think=False,
        temperature=0.0,
    )

    assert out == "hello"
    url, payload, _timeout = session.posts[0]
    assert url == "http://ollama:11434/api/chat"
    assert payload["stream"] is False
    assert payload["format"] == {"type": "object"}
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0


def test_chat_missing_message_returns_empty_string() -> None:
    session = FakeSession(FakeResponse({}))
    client = OllamaClient("http://x", session=session)
    assert client.chat("m", []) == ""


def test_generate_passes_images_and_returns_response() -> None:
    session = FakeSession(FakeResponse({"response": "a green bird"}))
    client = OllamaClient("http://x", session=session)

    out = client.generate("qwen2.5vl:7b", "describe", images=["BASE64"])

    assert out == "a green bird"
    _url, payload, _timeout = session.posts[0]
    assert payload["images"] == ["BASE64"]
    assert payload["prompt"] == "describe"


def test_is_available_reflects_reachability() -> None:
    assert OllamaClient("http://x", session=FakeSession(FakeResponse({}, ok=True))).is_available()
    assert not OllamaClient(
        "http://x", session=FakeSession(FakeResponse({}, ok=False))
    ).is_available()


def test_generate_serialises_vision_calls() -> None:
    import threading
    import time

    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    class ConcurrencyTrackingSession:
        def post(self, url, json, timeout):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return FakeResponse({"response": "ok"})

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=ConcurrencyTrackingSession(), vision_concurrency=1)
    threads = [threading.Thread(target=lambda: client.generate("m", "p", images=["B"])) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # With one slot, never more than one vision call hit the server at once.
    assert state["max"] == 1


def test_generate_allows_configured_concurrency() -> None:
    import threading
    import time

    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    class TrackingSession:
        def post(self, url, json, timeout):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return FakeResponse({"response": "ok"})

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=TrackingSession(), vision_concurrency=2)
    threads = [threading.Thread(target=lambda: client.generate("m", "p", images=["B"])) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] <= 2


def test_post_json_retries_transient_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    class FlakySession:
        def post(self, url, json, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse({}, ok=False, status=502)  # Olla backend busy
            return FakeResponse({"message": {"content": "ok"}})

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=FlakySession(), max_retries=2, retry_backoff_seconds=0)
    assert client.chat("m", []) == "ok"
    assert calls["n"] == 2  # one 502, one success


def test_post_json_does_not_retry_4xx() -> None:
    calls = {"n": 0}

    class ClientErrorSession:
        def post(self, url, json, timeout):
            calls["n"] += 1
            return FakeResponse({}, ok=False, status=400)  # bad request: won't recover

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=ClientErrorSession(), max_retries=3, retry_backoff_seconds=0)
    try:
        client.chat("m", [])
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("expected HTTPError to surface immediately")
    assert calls["n"] == 1  # a 4xx is not retried


def test_post_json_retries_transient_429() -> None:
    # 429 (rate limit) is a 4xx but transient — it must be retried, not fast-failed.
    calls = {"n": 0}

    class RateLimitedSession:
        def post(self, url, json, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse({}, ok=False, status=429)
            return FakeResponse({"message": {"content": "ok"}})

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=RateLimitedSession(), max_retries=2, retry_backoff_seconds=0)
    assert client.chat("m", []) == "ok"
    assert calls["n"] == 2  # one 429, one success


def test_post_json_raises_after_exhausting_retries() -> None:
    class DeadSession:
        def post(self, url, json, timeout):
            return FakeResponse({}, ok=False, status=502)

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=DeadSession(), max_retries=1, retry_backoff_seconds=0)
    try:
        client.generate("m", "p", images=["B"])
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("expected HTTPError after retries exhausted")


def test_chat_passes_num_predict_cap() -> None:
    session = FakeSession(FakeResponse({"message": {"content": "ok"}}))
    client = OllamaClient("http://x", session=session)
    client.chat("m", [{"role": "user", "content": "hi"}], num_predict=512)
    _url, payload, _timeout = session.posts[0]
    assert payload["options"]["num_predict"] == 512


def test_generate_passes_num_predict_cap() -> None:
    session = FakeSession(FakeResponse({"response": "ok"}))
    client = OllamaClient("http://x", session=session)
    client.generate("m", "p", num_predict=256)
    _url, payload, _timeout = session.posts[0]
    assert payload["options"]["num_predict"] == 256


def test_post_json_does_not_retry_read_timeout() -> None:
    # A ReadTimeout means the backend accepted the request and is still
    # generating; retrying spawns a SECOND full generation instead of cancelling
    # the first, tripling CPU on a busy backend. It must fail fast, not retry.
    calls = {"n": 0}

    class ReadTimeoutSession:
        def post(self, url, json, timeout):
            calls["n"] += 1
            raise requests.exceptions.ReadTimeout("generation too slow")

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=ReadTimeoutSession(), max_retries=3, retry_backoff_seconds=0)
    try:
        client.chat("m", [])
    except requests.exceptions.ReadTimeout:
        pass
    else:
        raise AssertionError("expected ReadTimeout to surface without retry")
    assert calls["n"] == 1  # not retried


def test_post_json_still_retries_connect_timeout() -> None:
    # A ConnectTimeout never reached the backend (nothing is generating), so a
    # retry is safe and routes around a briefly-unreachable Olla backend.
    calls = {"n": 0}

    class ConnectTimeoutThenOk:
        def post(self, url, json, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.ConnectTimeout("no route")
            return FakeResponse({"message": {"content": "ok"}})

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=ConnectTimeoutThenOk(), max_retries=2, retry_backoff_seconds=0)
    assert client.chat("m", []) == "ok"
    assert calls["n"] == 2  # connect failure retried, then succeeded


class FakeHealth:
    def __init__(self, *, available: bool = True, missing: tuple = ()) -> None:
        self.available = available
        self.successes = 0
        self.failures = 0
        self.model_missing: list[str] = []
        self.model_served: list[str] = []
        self.pokes = 0
        self._missing = set(missing)

    def has_model(self, name: str) -> bool:
        return name not in self._missing

    def note_success(self) -> None:
        self.successes += 1

    def note_failure(self) -> None:
        self.failures += 1

    def note_model_missing(self, model: str) -> None:
        self.model_missing.append(model)
        self._missing.add(model)

    def note_model_served(self, model: str) -> None:
        self.model_served.append(model)

    def poke(self) -> None:
        self.pokes += 1


def test_health_gate_blocks_chat_without_any_request() -> None:
    # A known-down backend must fail instantly — no connect attempt, no retry
    # ladder — so callers can queue work for backfill instead of hanging.
    from lib.ai.client import OllamaUnavailableError

    session = FakeSession(FakeResponse({"message": {"content": "hi"}}))
    client = OllamaClient("http://x", session=session, health=FakeHealth(available=False))
    try:
        client.chat("m", [])
    except OllamaUnavailableError:
        pass
    else:
        raise AssertionError("expected OllamaUnavailableError")
    assert session.posts == []  # never touched the network


def test_health_gate_blocks_generate_without_any_request() -> None:
    from lib.ai.client import OllamaUnavailableError

    session = FakeSession(FakeResponse({"response": "ok"}))
    client = OllamaClient("http://x", session=session, health=FakeHealth(available=False))
    try:
        client.generate("m", "p", images=["B"])
    except OllamaUnavailableError:
        pass
    else:
        raise AssertionError("expected OllamaUnavailableError")
    assert session.posts == []


def test_health_gate_open_lets_calls_through() -> None:
    # available=True is a no-op gate: the normal request path runs.
    session = FakeSession(FakeResponse({"message": {"content": "hello"}}))
    client = OllamaClient("http://x", session=session, health=FakeHealth(available=True))
    assert client.chat("m", [{"role": "user", "content": "hi"}]) == "hello"
    assert len(session.posts) == 1


def test_keep_alive_appears_in_chat_and_generate_payloads() -> None:
    # keep_alive stops the worker unloading the model between sporadic calls;
    # it must reach Ollama in BOTH endpoints' payloads.
    session = FakeSession(FakeResponse({"message": {"content": "ok"}, "response": "ok"}))
    client = OllamaClient("http://x", session=session, keep_alive="30m")
    client.chat("m", [])
    client.generate("m", "p")
    for _url, payload, _timeout in session.posts:
        assert payload["keep_alive"] == "30m"


def test_no_keep_alive_key_when_unset() -> None:
    # With keep_alive=None the key must be absent so Ollama's default applies.
    session = FakeSession(FakeResponse({"message": {"content": "ok"}, "response": "ok"}))
    client = OllamaClient("http://x", session=session)
    client.chat("m", [])
    client.generate("m", "p")
    for _url, payload, _timeout in session.posts:
        assert "keep_alive" not in payload


def test_health_note_success_called_on_200() -> None:
    # Every real 200 is recovery evidence for the health monitor.
    health = FakeHealth()
    session = FakeSession(FakeResponse({"message": {"content": "ok"}}))
    client = OllamaClient("http://x", session=session, health=health)
    client.chat("m", [])
    assert health.successes == 1
    assert health.failures == 0


def test_health_note_failure_on_connection_error() -> None:
    # A refused/unreachable connection is outage evidence.
    health = FakeHealth()

    class RefusedSession:
        def post(self, url, json, timeout):
            raise requests.exceptions.ConnectionError("refused")

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient(
        "http://x", session=RefusedSession(), health=health, max_retries=0, retry_backoff_seconds=0
    )
    try:
        client.chat("m", [])
    except requests.exceptions.ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError")
    assert health.failures == 1


def test_health_not_failed_on_read_timeout() -> None:
    # A ReadTimeout means the backend is reachable but slow — NOT an outage.
    health = FakeHealth()

    class SlowSession:
        def post(self, url, json, timeout):
            raise requests.exceptions.ReadTimeout("still generating")

        def get(self, url, timeout):
            return FakeResponse({})

    client = OllamaClient("http://x", session=SlowSession(), health=health, max_retries=0)
    try:
        client.chat("m", [])
    except requests.exceptions.ReadTimeout:
        pass
    else:
        raise AssertionError("expected ReadTimeout")
    assert health.failures == 0


def test_health_502_and_503_are_model_scoped_not_global() -> None:
    # Neither a 502 (one busy worker) nor a 503 (no healthy backend for THIS
    # request) may slam the GLOBAL gate: a cluster serving chat on a small
    # worker while the vision box is off would flap down/up on every vision
    # call. A 503 gates the requested model, pokes a probe (whose endpoint
    # counts decide whether it's really a full outage), and re-raises as the
    # ConnectionError subclass so outage handlers queue/pause instead of
    # burning give-up attempts. A 502 is one busy worker: retried, no gating.
    from lib.ai.client import OllamaUnavailableError

    health_502 = FakeHealth()
    client = OllamaClient(
        "http://x",
        session=FakeSession(FakeResponse({}, ok=False, status=502)),
        health=health_502,
        max_retries=0,
        retry_backoff_seconds=0,
    )
    try:
        client.chat("m", [])
    except requests.HTTPError:
        pass
    assert health_502.failures == 0
    assert health_502.model_missing == []

    health_503 = FakeHealth()
    session_503 = FakeSession(FakeResponse({}, ok=False, status=503))
    client = OllamaClient(
        "http://x",
        session=session_503,
        health=health_503,
        max_retries=2,
        retry_backoff_seconds=0,
    )
    try:
        client.chat("m", [])
    except OllamaUnavailableError:
        pass
    else:
        raise AssertionError("expected OllamaUnavailableError")
    assert health_503.failures == 0  # global gate untouched
    assert health_503.model_missing == ["m"]
    assert health_503.pokes == 1  # a probe settles whether it's a full outage
    assert len(session_503.posts) == 1  # no doomed retries after the verdict


def test_health_gate_blocks_model_nobody_serves() -> None:
    # Cluster up, but no healthy worker serves THIS model (the vision box is
    # off overnight): the call must fail instantly with the same
    # ConnectionError subclass — queued/backfilled, not retried into a 502.
    from lib.ai.client import OllamaUnavailableError

    session = FakeSession(FakeResponse({"response": "ok"}))
    client = OllamaClient(
        "http://x", session=session,
        health=FakeHealth(available=True, missing=("qwen2.5vl:7b",)),
    )
    try:
        client.generate("qwen2.5vl:7b", "describe", images=["B"])
    except OllamaUnavailableError:
        pass
    else:
        raise AssertionError("expected OllamaUnavailableError")
    assert session.posts == []  # never touched the network
    # Other models still flow normally.
    assert client.generate("qwen2.5vl:3b", "describe", images=["B"]) == "ok"


def test_404_marks_model_missing_and_raises_unavailable() -> None:
    # Olla answers 404 ("No ollama endpoints available") when nobody serves the
    # model: model-scoped evidence, no retry, and the raise is the SAME
    # ConnectionError subclass as a gated call — the router queues the message
    # and the backfill pauses WITHOUT burning a give-up attempt on the very
    # half-open re-test that discovered the model is still missing.
    from lib.ai.client import OllamaUnavailableError

    health = FakeHealth()
    session = FakeSession(FakeResponse({}, ok=False, status=404))
    client = OllamaClient(
        "http://x",
        session=session,
        health=health,
        max_retries=2,
        retry_backoff_seconds=0,
    )
    try:
        client.generate("qwen2.5vl:7b", "p", images=["B"])
    except OllamaUnavailableError:
        pass
    else:
        raise AssertionError("expected OllamaUnavailableError")
    assert health.model_missing == ["qwen2.5vl:7b"]
    assert health.failures == 0
    assert len(session.posts) == 1  # a 4xx never retries


def test_exhausted_5xx_ladder_does_not_mark_model_missing() -> None:
    # Ambiguous evidence must not trip the model breaker: an exhausted 502
    # ladder can be one overloaded worker, and a reproducible 500 can be a
    # single poison request (one bad photo crashing a runner). Gating the whole
    # model on either would stall every caller — the memory-backfill batch and
    # all live VLM features — for the TTL, on a backend that serves the model
    # fine. Only unambiguous no-backend-for-this-model verdicts (404/503) gate.
    for status in (502, 500):
        health = FakeHealth()
        client = OllamaClient(
            "http://x",
            session=FakeSession(FakeResponse({}, ok=False, status=status)),
            health=health,
            max_retries=1,
            retry_backoff_seconds=0,
        )
        try:
            client.chat("gemma3:12b", [])
        except requests.HTTPError:
            pass
        else:
            raise AssertionError("expected HTTPError")
        assert health.model_missing == []
        assert health.failures == 0


def test_success_reports_model_served() -> None:
    # Every 200 clears the model's missing marker (the half-open re-test that
    # succeeds is what reopens the model's gate).
    health = FakeHealth()
    client = OllamaClient(
        "http://x", session=FakeSession(FakeResponse({"message": {"content": "ok"}})),
        health=health,
    )
    client.chat("gemma3:4b", [])
    assert health.model_served == ["gemma3:4b"]


def test_post_timeout_is_connect_read_tuple() -> None:
    # A single timeout value would let a black-holed host hang CONNECT for the
    # whole generous read budget; connect is capped at 10s, read keeps the
    # requested value.
    session = FakeSession(FakeResponse({"message": {"content": "ok"}}))
    client = OllamaClient("http://x", session=session, timeout_seconds=120.0)
    client.chat("m", [])
    _url, _payload, timeout = session.posts[0]
    assert timeout == (10.0, 120.0)


def test_post_timeout_connect_not_above_requested_read() -> None:
    # A short per-call timeout also shortens the connect budget (min, not 10s).
    session = FakeSession(FakeResponse({"response": "ok"}))
    client = OllamaClient("http://x", session=session)
    client.generate("m", "p", timeout_seconds=4.0)
    _url, _payload, timeout = session.posts[0]
    assert timeout == (4.0, 4.0)


def test_generate_raises_busy_when_no_vision_slot_frees(monkeypatch) -> None:
    # The vision-slot wait is BOUNDED: with the single slot held by a stuck
    # generation, a second caller must fail with OllamaBusyError (making no HTTP
    # call) instead of blocking forever — and the slot accounting must survive.
    import threading

    from lib.ai.client import OllamaBusyError

    import pytest

    monkeypatch.setattr("lib.ai.client.VISION_SLOT_WAIT_SECONDS", 0.05)
    release = threading.Event()
    entered = threading.Event()

    class BlockingSession:
        def __init__(self) -> None:
            self.posts = 0

        def post(self, url, json, timeout):
            self.posts += 1
            entered.set()
            release.wait(5.0)  # the stuck in-flight generation
            return FakeResponse({"response": "a bird"})

    session = BlockingSession()
    client = OllamaClient("http://host:11434", session=session, vision_concurrency=1)
    first = threading.Thread(
        target=lambda: client.generate("vlm", "describe"), daemon=True
    )
    first.start()
    assert entered.wait(2.0)  # the slot is now held inside session.post

    try:
        with pytest.raises(OllamaBusyError):
            client.generate("vlm", "describe")  # can't get the slot in 0.05s
        assert session.posts == 1  # the busy caller never reached HTTP
    finally:
        release.set()
        first.join(timeout=5.0)

    # The slot was released properly by both the winner and the loser.
    assert client.generate("vlm", "describe") == "a bird"


class RecordingUsage:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []
        self.failures: list[str] = []

    def record(self, model, response):
        self.records.append((model, response))

    def record_failure(self, model):
        self.failures.append(model)


def test_usage_recorded_on_success_with_model_and_metering() -> None:
    # Every successful call reports its own metering to the usage recorder via
    # the single _post_json seam — chat and vision alike.
    usage = RecordingUsage()
    body = {"message": {"content": "hi"}, "prompt_eval_count": 12, "eval_count": 5}
    client = OllamaClient(
        "http://host:11434", session=FakeSession(FakeResponse(body)), usage=usage
    )
    client.chat("gemma3:4b", [{"role": "user", "content": "hey"}])
    assert usage.records == [("gemma3:4b", body)]
    assert usage.failures == []


def test_usage_failure_recorded_once_after_retries_exhaust() -> None:
    # A call that reaches the backend and dies counts ONE failure (not one per
    # retry attempt).
    usage = RecordingUsage()

    class FailingSession:
        def post(self, url, json, timeout):
            raise requests.exceptions.ConnectionError("refused")

    client = OllamaClient(
        "http://host:11434", session=FailingSession(),
        usage=usage, retry_backoff_seconds=0.0,
    )
    try:
        client.chat("gemma3:4b", [])
    except requests.RequestException:
        pass
    assert usage.failures == ["gemma3:4b"]
    assert usage.records == []


def test_usage_not_recorded_for_health_gated_calls() -> None:
    # A gated call never reached the backend — it must not count as a failure
    # (that would pollute the per-model failure stats during every outage).
    usage = RecordingUsage()

    class DownHealth:
        available = False

    client = OllamaClient(
        "http://host:11434", session=FakeSession(FakeResponse({})),
        usage=usage, health=DownHealth(),
    )
    try:
        client.chat("gemma3:4b", [])
    except requests.RequestException:
        pass
    assert usage.failures == [] and usage.records == []
