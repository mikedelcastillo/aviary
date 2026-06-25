from __future__ import annotations

from lib.ai.intent import (
    INTENT_SCHEMA,
    Intent,
    build_system_prompt,
    classify_intent,
    parse_intent,
)


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return self.content


def test_parse_intent_reads_action_and_argument() -> None:
    assert parse_intent('{"action": "find", "argument": "percy"}') == Intent("find", "percy")
    assert parse_intent('{"action": "pause", "argument": "10m"}') == Intent("pause", "10m")


def test_parse_intent_unknown_action_falls_back_to_chat() -> None:
    assert parse_intent('{"action": "launch_rocket", "argument": ""}') == Intent("chat", "")


def test_parse_intent_non_json_falls_back_to_chat() -> None:
    assert parse_intent("I think you want to pause") == Intent("chat", "")


def test_parse_intent_tolerates_missing_argument() -> None:
    assert parse_intent('{"action": "status"}') == Intent("status", "")


def test_parse_intent_handles_none_and_empty() -> None:
    # parse_intent is documented as total: a None/empty model reply must fall
    # back to chat, not raise while formatting the warning log.
    assert parse_intent(None) == Intent("chat", "")  # type: ignore[arg-type]
    assert parse_intent("") == Intent("chat", "")


def test_system_prompt_lists_birds() -> None:
    prompt = build_system_prompt(["percy", "matcha"])
    assert "percy" in prompt and "matcha" in prompt
    # The find-vs-history distinction must be spelled out.
    assert "what did percy do today" in prompt.lower()


def test_system_prompt_routes_care_questions_to_chat() -> None:
    # Care/safety/how-to questions should be steered to chat (care knowledge),
    # not the activity log, even when they name a bird.
    prompt = build_system_prompt(["percy"]).lower()
    assert "care knowledge" in prompt
    assert "avocado" in prompt  # the safe-vs-toxic example


def test_classify_intent_uses_structured_output() -> None:
    client = FakeClient('{"action": "find", "argument": "matcha"}')
    intent = classify_intent(client, "qwen3:4b", "where is matcha?", ["matcha"])
    assert intent == Intent("find", "matcha")
    call = client.calls[0]
    # Structured output + fast, deterministic settings for routing.
    assert call["fmt"] == INTENT_SCHEMA
    assert call["think"] is False
    assert call["temperature"] == 0.0
