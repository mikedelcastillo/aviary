from __future__ import annotations

from lib.ai.memory import ConversationMemory


def test_records_and_returns_history_per_chat() -> None:
    memory = ConversationMemory(max_messages=20)
    memory.record(1, "user", "hi")
    memory.record(1, "assistant", "hello!")
    memory.record(2, "user", "other chat")

    assert memory.history(1) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]
    assert memory.history(2) == [{"role": "user", "content": "other chat"}]


def test_history_is_capped_to_max() -> None:
    memory = ConversationMemory(max_messages=4)
    for i in range(10):
        memory.record(1, "user", str(i))
    history = memory.history(1)
    assert len(history) == 4
    assert [m["content"] for m in history] == ["6", "7", "8", "9"]


def test_empty_content_is_ignored() -> None:
    memory = ConversationMemory()
    memory.record(1, "assistant", "")
    assert memory.history(1) == []
