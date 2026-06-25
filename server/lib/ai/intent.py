"""Map a free-text Telegram message to one server action via the language model.

"stop the cams" -> pause, "where's percy?" -> find percy, "reload cams" ->
discover, and anything conversational -> chat (handled by the memory/Q&A layer).
We use Ollama's structured-output mode (a JSON schema) so the model returns a
clean ``{action, argument}`` instead of prose to parse.

The prompt building and response parsing are pure functions so they can be
tested without a model; :func:`classify_intent` ties them to a client.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from lib.ai.client import OllamaClient


LOGGER = logging.getLogger("lib.ai.intent")

# Every action the router can emit. "chat" is the catch-all for questions and
# conversation, routed to the memory/VLM layer rather than a command.
ACTIONS = (
    "pause", "resume", "find", "stop_find", "discover",
    "status", "snapshot", "activity", "chat",
)

# Ollama ``format`` schema: constrains the model to a valid action + argument.
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "argument": {"type": "string"},
    },
    "required": ["action", "argument"],
}


@dataclass(frozen=True)
class Intent:
    action: str
    argument: str = ""


def build_system_prompt(findable_birds: list[str]) -> str:
    birds = ", ".join(findable_birds) if findable_birds else "(unknown)"
    return (
        "You are the command router for a home bird-camera server (\"aviary\").\n"
        "Translate the user's message into exactly ONE action, returned as JSON.\n\n"
        "Actions:\n"
        '- "pause": stop the cameras / privacy mode / "stop recording" / "stop the cams". '
        'argument = a duration if given (e.g. "10m", "1 hour"), else "".\n'
        '- "resume": turn the cameras back on / play / resume / unpause. argument = "".\n'
        '- "find": locate one or more birds RIGHT NOW — "where is X", "find X", "is X out", '
        '"find the cockatiels", "find any bird". argument = exactly the bird(s) or group the '
        'user named, e.g. "percy", "percy and matcha", "the cockatiels", "lovebirds", '
        '"any bird". Groups (cockatiels, lovebirds, budgies, birds) and "any" are allowed.\n'
        '- "stop_find": cancel an in-progress search — "stop looking", "cancel the search", '
        '"never mind". argument = "". (But "look for X instead" is a NEW find, not stop_find.)\n'
        '- "discover": rescan / reload / refresh the cameras on the network. argument = "".\n'
        '- "status": how the cameras or system are doing, health, what is online. argument = "".\n'
        '- "snapshot": take or show pictures from all cameras right now. argument = "".\n'
        '- "activity": ANY question or request about what a bird IS or WAS doing, its day, '
        'behaviour, whereabouts-summary, or recent photos — including specific day-lookback '
        'questions. Examples: "what did percy do today", "what is draft up to", "what has '
        'pizza been up to", "what are the birds doing", "what\'s going on", "anything '
        'happening?", "how was matcha today", "did pizza eat today", "did matcha and jynx '
        'spend time together", "did the birds take a bath", "when did the birds go in their '
        'cage", "are the birds asleep", "what did jynx do this morning", "show me recent '
        'photos of pizza", "where are the birds", "where is everyone". argument = the '
        'bird(s)/group asked about (e.g. "pizza", "matcha and jynx"), or "" for all birds.\n'
        '- "chat": greetings, thanks, or anything that isn\'t about the birds\' whereabouts '
        'or activity. argument = "".\n\n'
        "Rules:\n"
        '- "where is percy" -> find (locate one specific bird). "where are the birds" / '
        '"where is everyone" -> activity (a summary of all, not a single-bird locate). '
        '"what did percy do today" / "what is percy up to" / "show me photos of percy" -> '
        'activity. "find the cockatiels" / "find any bird" -> find. "show me the cameras" / '
        '"take a snapshot" -> snapshot (live capture), but "show me photos of <bird>" -> '
        "activity (recent collected photos). Use find to locate, activity for "
        "behaviour/photos/summaries.\n"
        f"- Known birds: {birds}. Groups: cockatiels, lovebirds, budgies, birds. If a find "
        "target is not known, still put what they said in argument.\n"
        "- Output ONLY the JSON object."
    )


def build_messages(user_text: str, findable_birds: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(findable_birds)},
        {"role": "user", "content": user_text},
    ]


def parse_intent(content: str) -> Intent:
    """Parse the model's JSON into an :class:`Intent`, defaulting to chat.

    A non-JSON response, a missing/unknown action, or any shape surprise all
    fall back to ``chat`` so a malformed model reply never crashes routing.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        LOGGER.warning("Intent response was not JSON: %r", content[:200])
        return Intent("chat", "")
    if not isinstance(data, dict):
        return Intent("chat", "")
    action = str(data.get("action", "")).strip().lower()
    if action not in ACTIONS:
        action = "chat"
    argument = str(data.get("argument", "") or "").strip()
    return Intent(action, argument)


def classify_intent(
    client: OllamaClient,
    model: str,
    user_text: str,
    findable_birds: list[str],
) -> Intent:
    """Ask the language model to route ``user_text`` to a single action.

    ``think=False`` and ``temperature=0`` keep it fast and deterministic. Raises
    on a client/transport error so the caller can reply that the AI is offline.
    """
    content = client.chat(
        model,
        build_messages(user_text, findable_birds),
        fmt=INTENT_SCHEMA,
        think=False,
        temperature=0.0,
    )
    return parse_intent(content)
