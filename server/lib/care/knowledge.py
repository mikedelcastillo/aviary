"""Retrieve the right care facts for a question and format them for the LLM.

The chat / Q&A path is a generalist: it should answer "can the birds eat
avocado?", "how long do they sleep?", "is it too cold?", "why is Percy
plucking?" accurately rather than guessing. Rather than dump all of
:mod:`lib.care.facts` into every prompt (token cost, and it dilutes the signal),
this module pulls the handful of facts relevant to the message — by topic
keywords and by which species/birds are named — and renders a compact grounding
block. Pure chit-chat ("good morning!") matches nothing and gets no block.

All functions are pure and unit-tested; the live wiring lives in the chat path.
"""

from __future__ import annotations

import re

from lib.care.facts import (
    CARE_FACTS,
    SLEEP_HOURS_TEXT,
    TEMPERATURE_RANGE_TEXT,
    TOXIC_FOODS,
    CareFact,
    ToxicFood,
)

# The flock's species (so "the cockatiels" / "is it cold for budgies" resolve).
# Individuals -> species is owned by lib.roster; pass that map in to resolve a
# bird name ("percy") to its species here without duplicating the roster.
SPECIES = ("cockatiel", "lovebird", "budgie")

# Topic -> trigger words. A fact scores when its topic's words appear in the
# message; a fact is never shown for a topic the user didn't ask about.
TOPIC_KEYWORDS: dict[str, frozenset[str]] = {
    "diet": frozenset({
        "eat", "eats", "eating", "ate", "food", "foods", "feed", "feeding", "fed",
        "pellet", "pellets", "seed", "seeds", "diet", "veg", "vegetable", "vegetables",
        "veggies", "greens", "fruit", "fruits", "nutrition", "hungry", "snack", "snacks",
        "treat", "treats", "grit", "calcium", "cuttlebone", "supplement", "supplements",
        "water", "drink", "drinking",
    }),
    "toxic_foods": frozenset({
        "toxic", "poison", "poisonous", "dangerous", "danger", "safe", "harmful", "harm",
        "ok", "okay", "allowed", "give",
    }),
    "sleep_light": frozenset({
        "sleep", "sleeping", "asleep", "bed", "bedtime", "dark", "darkness", "light",
        "lights", "cover", "covered", "uncover", "night", "nighttime", "wake", "waking",
        "woke", "photoperiod", "nap", "rest", "resting", "sunrise", "sunset", "roost",
    }),
    "temperature": frozenset({
        "cold", "chilly", "chilled", "hot", "warm", "warmth", "temperature", "temp",
        "draft", "drafts", "drafty", "freezing", "heat", "heater", "weather", "degrees",
        "cool", "shivering",
    }),
    "enrichment": frozenset({
        "toy", "toys", "bored", "boredom", "play", "playing", "enrichment", "chew",
        "chewing", "exercise", "foraging", "forage", "shred", "stimulation", "fun",
        "entertain", "entertained",
    }),
    "social": frozenset({
        "lonely", "alone", "together", "aggressive", "aggression", "fight", "fighting",
        "bond", "bonded", "mate", "companion", "territorial", "hormonal", "nest",
        "nesting", "pair", "flock", "bully", "bullying", "attack", "attacking", "mixed",
        "house", "housing", "share", "sharing",
    }),
    "hygiene": frozenset({
        "clean", "cleaning", "bath", "bathe", "bathing", "dirty", "cage", "wash",
        "washing", "substrate", "liner", "hygiene", "mist", "dust", "dusty", "tidy",
    }),
    "health_signs": frozenset({
        "sick", "ill", "illness", "health", "healthy", "fluffed", "fluffy", "lethargic",
        "lethargy", "vet", "weight", "weigh", "symptom", "symptoms", "breathing",
        "breathe", "bobbing", "plucking", "molt", "molting", "droppings", "poop",
        "behavior", "behaviour", "wrong", "worried", "worry",
    }),
    "emergency": frozenset({
        "emergency", "dying", "die", "died", "blood", "bleeding", "egg-bound", "eggbound",
        "urgent", "hurt", "injured", "broken", "seizure", "collapsed",
    }),
}

_WORD_RE = re.compile(r"[a-z]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def detect_species(text: str, member_species: dict[str, str] | None = None) -> set[str]:
    """Species named in ``text`` — directly ("cockatiels") or via a bird name.

    ``member_species`` maps an individual ("percy") to its species ("lovebird");
    pass :func:`lib.roster`'s map so a name resolves without duplicating it here.
    """
    found: set[str] = set()
    tokens = _words(text)
    singulars = {_singular(token) for token in tokens}
    for species in SPECIES:
        if species in singulars:
            found.add(species)
    if member_species:
        for token in tokens:
            species = member_species.get(token)
            if species in SPECIES:
                found.add(species)  # type: ignore[arg-type]
    return found


def detected_topics(text: str) -> set[str]:
    """The care topics a message touches, by keyword."""
    tokens = _words(text)
    return {topic for topic, words in TOPIC_KEYWORDS.items() if tokens & words}


def toxic_food_in(text: str) -> ToxicFood | None:
    """The first toxic food named in ``text``, or None.

    Matched on word boundaries so "cola" never fires inside "chocolate" and
    "salt" never fires inside "salty-sounding" prose.
    """
    lowered = text.lower()
    for food in TOXIC_FOODS:
        for alias in food.aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return food
    return None


def relevant_facts(
    text: str,
    *,
    member_species: dict[str, str] | None = None,
    limit: int = 8,
) -> list[CareFact]:
    """The most relevant care facts for ``text`` (empty if it's not a care ask).

    Scored by topic-keyword hits, with a bonus when the fact's species was named
    and when a danger word pairs with a safety-critical fact. ``general`` facts
    are eligible for any matched topic; species facts only when that species (or
    one of its birds) is named, so "is it cold for the budgie" surfaces the
    budgie cold fact, not the lovebird one.
    """
    topics = detected_topics(text)
    if not topics:
        return []
    species = detect_species(text, member_species)
    tokens = _words(text)
    danger = bool(tokens & (TOPIC_KEYWORDS["emergency"] | {"toxic", "dangerous", "danger", "safe", "sick", "ill"}))

    scored: list[tuple[float, int, CareFact]] = []
    for index, fact in enumerate(CARE_FACTS):
        if fact.topic not in topics:
            continue
        # A species-specific fact only applies when that species is in play.
        if fact.species != "general" and fact.species not in species:
            continue
        score = float(len(TOPIC_KEYWORDS[fact.topic] & tokens))
        if fact.species in species:
            score += 2.0
        if fact.safety_critical and danger:
            score += 1.0
        if score <= 0:
            continue
        # Tie-break: safety-critical first, then original order (stable).
        scored.append((score, -index if not fact.safety_critical else 1000 - index, fact))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [fact for _, _, fact in scored[:limit]]


def care_context(
    text: str,
    *,
    member_species: dict[str, str] | None = None,
    limit: int = 8,
) -> str | None:
    """A compact grounding block of care facts for the LLM, or None.

    Returns None for messages that aren't about care (so casual chat stays
    cheap). When the message names a toxic food the answer leads with that, so a
    "can they eat avocado?" is answered correctly and unambiguously.
    """
    lines: list[str] = []

    food = toxic_food_in(text)
    if food is not None:
        lines.append(f"TOXIC — {food.name} is dangerous to these birds: {food.reason}.")

    facts = relevant_facts(text, member_species=member_species, limit=limit)
    for fact in facts:
        tag = " (safety-critical)" if fact.safety_critical else ""
        scope = "" if fact.species == "general" else f"[{fact.species}] "
        lines.append(f"- {scope}{fact.fact}{tag}")

    # A bare "is it bedtime?" / "is it cold?" can match a topic but few facts;
    # fold in the headline numbers so the model always has them to anchor on.
    topics = detected_topics(text)
    if "sleep_light" in topics:
        lines.append(f"- Sleep need: {SLEEP_HOURS_TEXT}")
    if "temperature" in topics:
        lines.append(f"- Temperature: {TEMPERATURE_RANGE_TEXT}")

    if not lines:
        return None
    header = (
        "Relevant bird-care knowledge (use it to answer accurately; never "
        "contradict a safety-critical line, and if the question is a health "
        "emergency, urge an avian vet):"
    )
    return header + "\n" + "\n".join(lines)
