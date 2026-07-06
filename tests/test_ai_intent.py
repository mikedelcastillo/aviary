from __future__ import annotations

from lib.ai.intent import (
    INTENT_SCHEMA,
    Intent,
    _correct_photo_snapshot,
    build_messages,
    build_system_prompt,
    classify_intent,
    parse_intent,
)


def test_photo_of_bird_reroutes_from_snapshot() -> None:
    birds = ["percy", "matcha", "bambi"]
    # A photo/picture request naming a bird (or "birds") is a saved-photo search,
    # not a live snapshot — even when the small model labeled it snapshot.
    assert _correct_photo_snapshot(Intent("snapshot", "percy"),
                                   "show me a picture of percy taking a bath", birds).action == "activity"
    assert _correct_photo_snapshot(Intent("snapshot", ""),
                                   "show me photos of birds eating", birds).action == "activity"
    # Real snapshots (no photo-of-a-bird phrasing) are left alone.
    assert _correct_photo_snapshot(Intent("snapshot", ""), "take a snapshot", birds).action == "snapshot"
    assert _correct_photo_snapshot(Intent("snapshot", ""), "show me the cameras", birds).action == "snapshot"
    # Non-snapshot intents pass through untouched.
    assert _correct_photo_snapshot(Intent("find", "percy"), "find percy", birds).action == "find"


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


def test_parse_intent_accepts_sleep_action() -> None:
    assert parse_intent('{"action": "sleep", "argument": "week"}') == Intent("sleep", "week")


def test_parse_intent_rejects_removed_care_action() -> None:
    # The 'care' feature was removed, so 'care' is no longer a valid action and an
    # intent naming it falls back to chat (where the bot defers care questions).
    assert parse_intent('{"action": "care", "argument": "diet"}').action == "chat"


def test_parse_intent_accepts_weather_action() -> None:
    assert parse_intent('{"action": "weather", "argument": ""}') == Intent("weather", "")


def test_system_prompt_routes_forecast_questions_to_weather() -> None:
    prompt = build_system_prompt(["percy"]).lower()
    # The forecast phrasings the user expects must be present so the model routes
    # them to "weather" rather than the care-knowledge "chat".
    assert '"weather"' in prompt
    assert "will it rain" in prompt
    assert "how hot will it be later" in prompt
    assert "will it be cold later" in prompt
    # And the care/forecast boundary is spelled out.
    assert "is it too cold for them" in prompt


def test_system_prompt_has_no_care_action() -> None:
    # The care guide/action is gone; the prompt must not offer a "care" action.
    prompt = build_system_prompt(["percy"]).lower()
    assert "care guide" not in prompt
    assert '"care"' not in prompt


def test_system_prompt_mentions_sleep_routing() -> None:
    prompt = build_system_prompt(["percy"]).lower()
    assert "sleep score" in prompt and "night fright" in prompt


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
    # Care/safety/how-to questions are conversation (chat), not the activity log,
    # even when they name a bird — the bot defers them.
    prompt = build_system_prompt(["percy"]).lower()
    assert "avocado" in prompt  # the safe-vs-toxic example still cues "chat"
    assert '"chat"' in prompt


def test_classify_intent_uses_structured_output() -> None:
    client = FakeClient('{"action": "find", "argument": "matcha"}')
    intent = classify_intent(client, "qwen3:4b", "where is matcha?", ["matcha"])
    assert intent == Intent("find", "matcha")
    call = client.calls[0]
    # Structured output + fast, deterministic settings for routing.
    assert call["fmt"] == INTENT_SCHEMA
    assert call["think"] is False
    assert call["temperature"] == 0.0


def test_build_messages_without_prior_passes_text_verbatim() -> None:
    msgs = build_messages("what about percy?", ["percy"])
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "what about percy?"


def test_build_messages_with_prior_adds_followup_context() -> None:
    # A short follow-up must carry the previous message + its action so the model
    # can continue it ("did draft fly?" -> activity) rather than default to find.
    msgs = build_messages(
        "what about percy?", ["percy"], prior=("did draft or pizza fly today?", "activity")
    )
    user = msgs[-1]["content"]
    assert "did draft or pizza fly today?" in user
    assert "activity" in user
    assert "what about percy?" in user  # the actual message to classify is still present


def test_classify_intent_threads_prior_into_messages() -> None:
    client = FakeClient('{"action": "activity", "argument": "percy"}')
    classify_intent(
        client, "gemma3:4b", "what about percy?", ["percy"],
        prior=("did draft fly today?", "activity"),
    )
    sent_user = client.calls[0]["messages"][-1]["content"]
    assert "did draft fly today?" in sent_user
