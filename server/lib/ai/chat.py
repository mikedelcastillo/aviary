"""Conversational replies for messages that aren't a command.

Kept separate from intent routing so both ``main`` (the live bot) and the
``llm-harness`` test tool share one definition of the caretaker persona and one
reply path. Feature 4 layers conversation history + activity context on top by
passing ``history`` and extending the system prompt.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from lib.ai.client import OllamaClient
from lib.textfmt import flatten_tables


LOGGER = logging.getLogger("lib.ai.chat")

# Shown when the model only ever returns garbage (see ``looks_degenerate``).
CHAT_FALLBACK = "Sorry — I garbled that one. Mind asking me again?"

# A chat turn is interactive — the user is watching "typing…". 90s covers a
# cold model load plus a full 384-token reply with room to spare; past that the
# backend is wedged and failing fast lets the message be queued for replay.
CHAT_TIMEOUT_SECONDS = 90.0

# Strips an inline <think>...</think> block some models/Ollama versions emit.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_BOXED_FINAL = re.compile(r"\\boxed\{([^{}]+)\}")


def strip_thinking(text: str) -> str:
    """Remove any leaked chain-of-thought so only the final answer is shown.

    We run chat with ``think=True`` so Ollama returns reasoning in a SEPARATE
    field and ``content`` is already clean — but this is a cheap belt-and-braces
    guard against a model that inlines ``<think>`` tags anyway.
    """
    return _THINK_BLOCK.sub("", text).strip()


def looks_degenerate(text: str) -> bool:
    """True when a reply is repetition garbage rather than real content.

    Small models occasionally fall into a loop and emit a wall of one symbol
    (``@@@@@@@@``), a single character dominating the message, or one short token
    over and over. That must never reach the user, so the caller falls back.

    We flag a reply only when the junk DOMINATES it — a long reply that merely
    contains a divider like ``=====`` keeps its real content rather than being
    thrown away wholesale.
    """
    s = text.strip()
    if not s:
        return False  # empty is "no content", handled separately by the caller
    no_space = re.sub(r"\s+", "", s)
    # Entirely one repeated ASCII symbol — the classic "@@@@@@@" / "------" dump.
    if len(no_space) >= 4 and len(set(no_space)) == 1:
        ch = no_space[0]
        if ch.isascii() and not ch.isalnum():
            return True
    # One non-alphanumeric symbol dominates a long reply.
    if len(no_space) >= 16:
        ch, count = Counter(no_space).most_common(1)[0]
        if not ch.isalnum() and count / len(no_space) > 0.5:
            return True
    # One short token repeated over and over ("bird bird bird ...").
    tokens = s.split()
    if len(tokens) >= 8:
        tok, count = Counter(tokens).most_common(1)[0]
        if len(tok) <= 15 and count / len(tokens) > 0.6:
            return True
    return False


def clean_reply(text: str) -> str:
    """Post-process raw model text into something safe to send, or ``""``.

    Strips leaked thinking, flattens markdown tables Telegram can't render, and
    returns an EMPTY string when the output is degenerate (so callers fall back
    to a template or a friendly retry message instead of forwarding garbage).
    """
    cleaned = flatten_tables(strip_thinking(text))
    boxed = _BOXED_FINAL.findall(cleaned)
    if boxed:
        cleaned = boxed[-1].strip()
    if looks_degenerate(cleaned):
        return ""
    return cleaned.strip()

# The caretaker persona for free-text chit-chat. It is a MONITORING assistant: it
# reports what the cameras see and makes small talk, but does NOT dispense bird-care
# advice (that feature was removed) and must NOT invent specific bird activities
# (that's the memory/VLM layer's job). Written as short, distinct rules — an instruct
# model (gemma3) follows a bulleted contract far more reliably than one buried
# paragraph, which keeps it terse and stops it leaking the species name or the prompt.
CHAT_SYSTEM_PROMPT = (
    "You are the warm caretaker of a home aviary, watching six pet birds over several "
    "cameras. Their names and pronouns: Percy (she), Matcha (he), Jynx (he), Bambi "
    "(she), Draft (he), Pizza (he).\n\n"
    "Reply rules:\n"
    "- Answer directly and warmly. Give ONLY the answer — never restate the question, "
    "quote these instructions, prefix a label or header, or narrate your reasoning.\n"
    "- You are talking to the birds' OWNER (a person), never to a bird. Do NOT open with "
    "or use a bird's name as a greeting or vocative (no \"Hello, Percy\", no \"Percy, it's "
    "daytime\", no \"Goodnight, Percy\"); address the owner as \"you\", or use no name. A "
    "bird's name may appear only as the SUBJECT of a statement about that bird.\n"
    "- Keep it to ONE sentence under 35 words.\n"
    "- Call each bird by NAME with its correct pronoun. NEVER add its species or breed "
    "(never \"Percy the lovebird\", never \"the budgie Bambi\").\n"
    "- You may get a 'Current aviary state' block; use it to say what is happening right "
    "now (time, day/night, who is visible, camera health). Never invent a sighting or a "
    "specific thing a bird did that is not in that block; if you do not know, say so.\n"
    "- When the user asks WHERE a specific bird is and it is not in the state, offer to "
    "look and suggest they say \"find <bird>\".\n"
    "- You are a MONITORING assistant, NOT a vet. For any bird-care, health, diet, "
    "feeding, or safe/toxic-food question, do NOT give advice — briefly say you are set "
    "up to watch the birds rather than advise on their care, and suggest an avian vet for "
    "a health concern."
)


def build_chat_messages(
    text: str,
    history: list[dict[str, str]] | None = None,
    *,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
    context: str | None = None,
) -> list[dict[str, str]]:
    """Assemble chat messages: system (+ optional context), history, then turn.

    ``context`` is extra grounding (e.g. recent sightings) appended to the system
    message by the memory layer; ``history`` is prior turns for continuity.
    """
    system = system_prompt if not context else f"{system_prompt}\n\n{context}"
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": text})
    return messages


def chat_reply(
    client: OllamaClient,
    model: str,
    text: str,
    history: list[dict[str, str]] | None = None,
    *,
    context: str | None = None,
) -> str:
    """Generate a conversational reply. Raises on transport error (caller degrades).

    Runs with ``think=False``: ``llm_model`` is an INSTRUCT model (gemma3), which
    answers directly with no chain-of-thought, so a modest ``num_predict`` bounds
    the reply without risk. (A *reasoning* model like qwen3 is the wrong fit here —
    with ``think=True`` it spends the whole ``num_predict`` budget deliberating and
    returns EMPTY content, tripping the fallback below; with ``think=False`` it
    dumps that reasoning into ``content``. An instruct model sidesteps both.)

    The raw reply is run through :func:`clean_reply` (strip any stray thinking,
    flatten tables, reject degenerate ``@@@@`` loops). A degenerate/empty result is
    retried ONCE — usually a sampling fluke a fresh draw clears — and only if that
    also fails does it return the friendly :data:`CHAT_FALLBACK`.
    """
    messages = build_chat_messages(text, history, context=context)
    for _ in range(2):
        reply = clean_reply(
            client.chat(
                model, messages, think=False, num_predict=384,
                timeout_seconds=CHAT_TIMEOUT_SECONDS,
            )
        )
        if reply:
            return reply
    return CHAT_FALLBACK
