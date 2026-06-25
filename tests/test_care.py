from __future__ import annotations

from lib.care import (
    CARE_FACTS,
    SLEEP_HOURS,
    TOXIC_FOODS,
    care_context,
    care_reply,
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


def test_toxic_food_detection_handles_plurals() -> None:
    # Plural mentions must still trip the safety gate — this is the toxic-food
    # guard, so a miss is fail-open (the bot might call a toxic food fine).
    assert toxic_food_in("can the birds eat scallions").name == "onion"
    assert toxic_food_in("are leeks safe for percy").name == "onion"
    assert toxic_food_in("can they have shallots").name == "onion"
    assert toxic_food_in("can i give them beers").name == "alcohol"
    # Only the PIT is toxic — bare fruit flesh (cherries/peaches) is safe, so a
    # bare fruit name must NOT trip the gate (it would hijack benign questions).
    assert toxic_food_in("are cherry pits safe").name == "fruit pits"
    assert toxic_food_in("are cherries safe") is None
    assert toxic_food_in("what about peaches") is None


def test_budgie_avocado_fact_surfaces_for_natural_phrasing() -> None:
    # "can the budgie eat avocado" detects only the 'diet' topic, but naming a
    # toxic food must still surface the species-specific avocado fact.
    facts = relevant_facts("can the budgie eat avocado", member_species=MEMBER_SPECIES)
    assert any(f.species == "budgie" and f.topic == "toxic_foods" for f in facts)


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


def test_care_reply_overview_when_empty() -> None:
    text = care_reply("")
    assert "Caring for the flock" in text
    assert "/care diet" in text  # points to the deeper topics


def test_care_reply_toxic_lists_foods() -> None:
    text = care_reply("toxic")
    assert "keep AWAY" in text
    assert "Avocado" in text and "Chocolate" in text


def test_care_reply_named_toxic_food_leads_with_warning() -> None:
    text = care_reply("avocado")
    assert text.startswith("⚠️ Avocado is dangerous")


def test_care_reply_topic_returns_relevant_facts() -> None:
    text = care_reply("sleep")
    assert "Sleep & light" in text
    low, high = SLEEP_HOURS
    assert f"{low}-{high}" in text  # the headline sleep numbers
    assert "•" in text  # bulleted facts


def test_care_reply_species_returns_that_species_facts() -> None:
    text = care_reply("cockatiel")
    assert "[cockatiel]" in text  # this species' specific facts surface
    assert "[lovebird]" not in text and "[budgie]" not in text  # not other species' specifics


def test_care_reply_unknown_query_gives_hint() -> None:
    text = care_reply("xyzzy")
    assert "don't have specific care notes" in text


def test_care_reply_bird_name_wins_over_keyword_collision() -> None:
    # "draft" is a cockatiel AND a temperature keyword — the bird must win.
    text = care_reply("draft", member_species=MEMBER_SPECIES)
    assert "[cockatiel]" in text  # the cockatiel profile, not generic temperature


def test_care_reply_permission_question_is_not_a_toxic_warning() -> None:
    # Grapes are safe — "can I give them grapes" must not dump the toxic list.
    assert "keep AWAY" not in care_reply("can i give them grapes")


def test_care_reply_safe_is_not_the_toxic_list() -> None:
    assert "keep AWAY" not in care_reply("safe")
