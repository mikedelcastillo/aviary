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
    CARE_SUMMARY,
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
        # Hard danger words only — generic permission/"safe" words are deliberately
        # excluded so "can I give them grapes" / "/care safe" don't read as toxic.
        "toxic", "poison", "poisonous", "dangerous", "danger", "harmful", "harm",
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
            # Allow a simple trailing-"s" plural so a plural mention still trips
            # the safety gate ("scallions", "leeks", "beers"). Irregular plurals
            # (cherries, peaches) are listed explicitly in the alias data.
            if re.search(rf"\b{re.escape(alias)}s?\b", lowered):
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
    # Naming a toxic food implies the toxic_foods topic even when the phrasing is
    # just "can the budgie eat avocado" (topic 'diet' only) — so the curated,
    # species-specific toxic fact (e.g. budgie avocado dose) can surface.
    if toxic_food_in(text) is not None:
        topics = topics | {"toxic_foods"}
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


# -- /care command formatting ------------------------------------------------

# Topics a user can ask for by name, mapped to a friendly label for the header.
_TOPIC_LABELS = {
    "diet": "Diet & feeding",
    "toxic_foods": "Toxic foods",
    "sleep_light": "Sleep & light",
    "temperature": "Temperature",
    "enrichment": "Enrichment",
    "social": "Social & housing",
    "hygiene": "Hygiene & cleaning",
    "health_signs": "Health signs",
    "emergency": "Emergencies",
}


def care_overview() -> str:
    """The bare ``/care`` reply: the flock summary + how to dig deeper."""
    return (
        "🐦 Caring for the flock\n"
        + CARE_SUMMARY
        + "\n\nAsk for more: /care diet · /care sleep · /care temperature · /care health · "
        "/care toxic · or a bird/species (e.g. /care cockatiel, /care percy)."
    )


def toxic_food_list() -> str:
    """The full toxic-food list, for ``/care toxic``."""
    lines = ["⚠️ Foods to keep AWAY from the birds:"]
    for food in TOXIC_FOODS:
        lines.append(f"• {food.name.title()} — {food.reason}")
    return "\n".join(lines)


def care_reply(query: str, *, member_species: dict[str, str] | None = None, limit: int = 10) -> str:
    """Answer a ``/care`` request: an overview, a toxic list, or topic/species facts."""
    q = (query or "").strip()
    if not q:
        return care_overview()
    lower = q.lower()

    food = toxic_food_in(lower)
    species = detect_species(lower, member_species)
    topics = detected_topics(lower)
    # A bird NAME that collides with a topic keyword ("draft" the cockatiel vs the
    # temperature word) must not pull in that topic — recompute topics with the
    # known names dropped, so a bare name resolves to that species' care.
    if member_species:
        names = {token for token in _words(lower) if token in member_species}
        if names:
            topics = detected_topics(" ".join(t for t in _words(lower) if t not in names))

    # Explicit "show me the toxic foods" -> the full list (only on a hard danger
    # word, never on bare permission words like "give"/"ok"/"safe").
    if food is None and any(
        word in lower for word in ("toxic", "poison", "unsafe", "dangerous", "what can't", "can't eat", "cannot eat", "avoid", "bad food")
    ):
        return toxic_food_list()

    # A named toxic food leads with its specific warning; if nothing else matched,
    # that warning IS the answer (e.g. /care avocado).
    head: list[str] = []
    if food is not None:
        head.append(f"⚠️ {food.name.title()} is dangerous: {food.reason}.")

    matched: list[CareFact] = []
    for fact in CARE_FACTS:
        if topics and fact.topic not in topics:
            continue
        if species and fact.species != "general" and fact.species not in species:
            continue
        if not topics and not species:
            continue
        matched.append(fact)

    if not matched:
        if head:
            return "\n".join(head)
        return (
            f'I don\'t have specific care notes for "{q}". Try /care diet, /care sleep, '
            "/care temperature, /care health, /care toxic, or a bird/species name."
        )

    # Species-specific facts first, then general; safety-critical first within each.
    matched.sort(key=lambda f: (0 if (species and f.species in species) else 1, 0 if f.safety_critical else 1))

    parts = list(head)
    label = next((_TOPIC_LABELS[t] for t in _TOPIC_LABELS if t in topics), None)
    if species and not label:
        label = ", ".join(sorted(s.title() for s in species)) + " care"
    if label:
        parts.append(f"🐦 {label}")
    # Headline numbers — only when a matched fact doesn't already state them.
    shown = matched[:limit]
    if "sleep_light" in topics and not any("10-12" in f.fact for f in shown):
        parts.append(f"• {SLEEP_HOURS_TEXT}")
    if "temperature" in topics and not any("65-80" in f.fact for f in shown):
        parts.append(f"• {TEMPERATURE_RANGE_TEXT}")
    for fact in shown:
        scope = "" if fact.species == "general" else f"[{fact.species}] "
        tag = " ⚠️" if fact.safety_critical else ""
        parts.append(f"• {scope}{fact.fact}{tag}")
    return "\n".join(parts)


def care_tip(index: int) -> str:
    """A single rotating care tip (by ``index``, e.g. the ISO week) for a digest."""
    fact = CARE_FACTS[index % len(CARE_FACTS)]
    scope = "" if fact.species == "general" else f"({fact.species}) "
    flag = " ⚠️" if fact.safety_critical else ""
    return f"💡 Care tip of the week: {scope}{fact.fact}{flag}"
