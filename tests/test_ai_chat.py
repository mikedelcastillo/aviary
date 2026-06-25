from __future__ import annotations

from lib.ai.chat import build_chat_messages, strip_thinking


def test_strip_thinking_removes_inline_block() -> None:
    text = "<think>let me reason about this</think>Percy is by the window."
    assert strip_thinking(text) == "Percy is by the window."


def test_strip_thinking_passes_clean_text_through() -> None:
    assert strip_thinking("  Percy is napping.  ") == "Percy is napping."


def test_build_chat_messages_orders_system_history_then_user() -> None:
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]
    messages = build_chat_messages("where is percy?", history)
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "where is percy?"}


def test_build_chat_messages_appends_context_to_system() -> None:
    messages = build_chat_messages("hi", context="Recently seen: Percy on the perch.")
    assert "Recently seen: Percy on the perch." in messages[0]["content"]
