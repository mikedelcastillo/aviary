from __future__ import annotations

from lib.ai.chat import (
    CHAT_FALLBACK,
    CHAT_SYSTEM_PROMPT,
    build_chat_messages,
    chat_reply,
    clean_reply,
    flatten_tables,
    looks_degenerate,
    strip_thinking,
)


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


def test_chat_prompt_keeps_replies_short() -> None:
    prompt = CHAT_SYSTEM_PROMPT.lower()
    assert "one sentence" in prompt
    assert "under 35 words" in prompt
    assert "at most 3 compact bullets" in prompt


# -- markdown tables: Telegram sends plain text, so a GFM pipe table renders as
#    raw "| --- |" noise. We flatten it into readable lines. -------------------


def test_flatten_tables_pairs_header_with_each_row() -> None:
    text = (
        "Here's the flock:\n"
        "| Bird | Status |\n"
        "| --- | --- |\n"
        "| Percy | napping |\n"
        "| Matcha | active |\n"
        "Hope that helps!"
    )
    out = flatten_tables(text)
    assert "|" not in out
    assert "---" not in out
    assert "Bird: Percy, Status: napping" in out
    assert "Bird: Matcha, Status: active" in out
    # Surrounding prose is preserved.
    assert out.startswith("Here's the flock:")
    assert out.rstrip().endswith("Hope that helps!")


def test_flatten_tables_header_only_joins_its_cells() -> None:
    # A header + separator with no data rows: just join the header cells.
    text = "| Percy | Matcha | Jynx |\n|---|---|---|"
    out = flatten_tables(text)
    assert "|" not in out
    assert "Percy — Matcha — Jynx" in out


def test_flatten_tables_ragged_row_joins_when_columns_mismatch() -> None:
    # A data row whose cell count doesn't match the header can't be paired, so
    # its cells are joined plainly rather than mis-labelled.
    text = "| A | B |\n|---|---|\n| only-one-cell |"
    out = flatten_tables(text)
    assert "|" not in out
    assert "only-one-cell" in out


def test_flatten_tables_leaves_a_lone_pipe_alone() -> None:
    # A stray pipe in prose is NOT a table (no separator row) — leave it untouched.
    text = "Use the /find command | it locates a bird."
    assert flatten_tables(text) == text


# -- degenerate output: the model occasionally loops and emits "@@@@@@@@" or a
#    single token over and over. That garbage must never reach the user. -------


def test_looks_degenerate_catches_repeated_symbol_dump() -> None:
    assert looks_degenerate("@@@@@@@")
    assert looks_degenerate("----------------")
    assert looks_degenerate("######## ######## ########")


def test_looks_degenerate_catches_repeated_token() -> None:
    assert looks_degenerate("bird bird bird bird bird bird bird bird bird")


def test_looks_degenerate_passes_normal_replies() -> None:
    assert not looks_degenerate("Percy is napping on her favourite perch right now. 😴")
    assert not looks_degenerate("Yes! Broccoli and carrots are great for them.")
    assert not looks_degenerate("Hmm... let me think — I don't see Pizza right now.")
    assert not looks_degenerate("")  # empty is "no content", handled separately
    assert not looks_degenerate("😴")  # a lone emoji is fine


def test_looks_degenerate_keeps_a_reply_with_a_divider() -> None:
    # A real reply that merely CONTAINS a rule of dashes/equals — even a long one
    # — is not garbage; dropping the whole message over a divider is a false pos.
    assert not looks_degenerate("Status:\n--------\nPercy is fine, everyone's settled.")
    assert not looks_degenerate(
        "Here's the rundown:\n" + "=" * 30 + "\nPercy, Matcha and Jynx are all on the perch."
    )


def test_clean_reply_returns_empty_for_degenerate() -> None:
    assert clean_reply("<think>oops</think>@@@@@@@@@@@@") == ""


def test_clean_reply_strips_thinking_and_flattens_tables() -> None:
    out = clean_reply("<think>x</think>| A | B |\n|---|---|\n| 1 | 2 |")
    assert "think" not in out and "|" not in out
    assert "A: 1, B: 2" in out


def test_clean_reply_uses_boxed_final_answer_only() -> None:
    out = clean_reply("The user likely asks where Percy was. Therefore: \\boxed{near the curtains}")
    assert out == "near the curtains"


class _ScriptedClient:
    """Returns successive canned replies from ``replies`` on each chat() call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):
        self.calls += 1
        return self._replies.pop(0) if self._replies else ""


def test_chat_reply_retries_once_on_degenerate_then_succeeds() -> None:
    client = _ScriptedClient(["@@@@@@@@@@@@", "Percy is on the perch."])
    out = chat_reply(client, "m", "where is percy?")
    assert out == "Percy is on the perch."
    assert client.calls == 2  # garbage first, retried, got a real answer


def test_chat_reply_falls_back_when_both_attempts_degenerate() -> None:
    client = _ScriptedClient(["@@@@@@@@@@@@", "############"])
    out = chat_reply(client, "m", "hi")
    assert out == CHAT_FALLBACK
    assert client.calls == 2  # one retry, no more


def test_chat_reply_flattens_a_table_in_a_real_answer() -> None:
    client = _ScriptedClient(["| Bird | Where |\n|---|---|\n| Percy | perch |"])
    out = chat_reply(client, "m", "where is everyone?")
    assert "|" not in out
    assert "Bird: Percy, Where: perch" in out
    assert client.calls == 1
