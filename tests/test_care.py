from __future__ import annotations

from lib.care import (
    CARE_FACTS,
    SLEEP_HOURS,
    TOXIC_FOODS,
    care_context,
    detect_species,
    detected_topics,
    relevant_facts,
    toxic_food_in,
)

# A small individual -> species map (the live one comes from lib.roster).
MEMBER_SPECIES = {
    "percy": "lovebird", "matcha": "lovebird", "jynx": "lovebird",
    "bambi": "budgie", "draft": "cockatiel", "pizza": "cockatiel",
}


def test_knowledge_base_is_well_formed() -> None:
    assert len(CARE_FACTS) >= 30
    for fact in CARE_FACTS:
        assert fact.species in {"general", "cockatiel", "lovebird", "budgie"}
        assert fact.fact.strip()
        assert isinstance(fact.safety_critical, bool)
    # Some facts must be flagged safety-critical (toxic food, temperature, etc.).
    assert any(fact.safety_critical for fact in CARE_FACTS)


def test_toxic_food_detection() -> None:
    assert toxic_food_in("can the birds eat avocado?").name == "avocado"
    assert toxic_food_in("is guacamole ok for percy").name == "avocado"
    assert toxic_food_in("can they have some chocolate").name == "chocolate"
    assert toxic_food_in("a little garlic bread?").name == "garlic"
    # Safe foods and plain chat return nothing.
    assert toxic_food_in("can they eat broccoli and carrots") is None
    assert toxic_food_in("good morning friend") is None


def test_toxic_alias_respects_word_boundaries() -> None:
    # "cola" is a substring of "chocolate" but must not match as caffeine/soda,
    # and chocolate itself is correctly flagged.
    assert toxic_food_in("they love chocolate").name == "chocolate"
    # "salt" must not fire inside an unrelated word.
    assert toxic_food_in("we should asphalt the driveway") is None


def test_detect_species_directly_and_via_bird_name() -> None:
    assert detect_species("find the cockatiels") == {"cockatiel"}
    assert detect_species("is percy cold", MEMBER_SPECIES) == {"lovebird"}
    assert detect_species("how are draft and bambi", MEMBER_SPECIES) == {"cockatiel", "budgie"}
    # No species word and no known bird -> nothing.
    assert detect_species("how are the birds") == set()


def test_detected_topics() -> None:
    assert "diet" in detected_topics("what should they eat")
    assert "sleep_light" in detected_topics("is it bedtime yet")
    assert "temperature" in detected_topics("is it too cold in here")
    assert detected_topics("good morning") == set()


def test_relevant_facts_filters_by_species() -> None:
    # A budgie cold question surfaces the budgie temperature fact, not lovebird's.
    facts = relevant_facts("is it too cold for the budgie", member_species=MEMBER_SPECIES)
    species = {f.species for f in facts}
    assert "budgie" in species
    assert "lovebird" not in species  # other species' specifics are excluded
    # Pure chit-chat yields no facts.
    assert relevant_facts("good morning!") == []


def test_care_context_leads_with_toxic_warning() -> None:
    context = care_context("can the birds eat avocado?", member_species=MEMBER_SPECIES)
    assert context is not None
    assert "TOXIC" in context and "avocado" in context.lower()


def test_care_context_includes_sleep_numbers() -> None:
    context = care_context("how long should the birds sleep at night?")
    assert context is not None
    low, high = SLEEP_HOURS
    assert f"{low}-{high}" in context


def test_care_context_none_for_chitchat() -> None:
    assert care_context("hey there, how's it going?") is None


def test_toxic_foods_cover_the_critical_ones() -> None:
    names = {food.name for food in TOXIC_FOODS}
    for required in ("avocado", "chocolate", "caffeine", "alcohol", "onion"):
        assert required in names
