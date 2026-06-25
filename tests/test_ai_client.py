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
            raise requests.HTTPError(f"status {self.status_code}")


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
