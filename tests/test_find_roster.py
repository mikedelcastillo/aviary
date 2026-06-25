from __future__ import annotations

from lib.roster import (
    DEFAULT_SPECIES_MEMBERS,
    expand_targets,
    load_sexes,
    pronoun_map,
    pronoun_sentence,
)


# The live model's findable labels (what the server detects).
FINDABLE = [
    "bambi", "budgie", "cockatiel", "draft", "jynx",
    "lovebird", "matcha", "percy", "pizza", "unknown_bird",
]


def test_individual_name_resolves_to_itself() -> None:
    assert expand_targets("percy", FINDABLE) == ["percy"]
    assert expand_targets("Percy", FINDABLE) == ["percy"]


def test_strips_articles_and_filler() -> None:
    assert expand_targets("find the percy please", FINDABLE) == ["percy"]
    assert expand_targets("where is matcha", FINDABLE) == ["matcha"]


def test_species_group_expands_to_members_plus_outline() -> None:
    # "cockatiels" -> the individuals (draft, pizza) + the IR species outline.
    result = expand_targets("the cockatiels", FINDABLE)
    assert set(result) == {"draft", "pizza", "cockatiel"}


def test_lovebirds_group() -> None:
    result = expand_targets("lovebirds", FINDABLE)
    assert set(result) == {"percy", "matcha", "jynx", "lovebird"}


def test_budgies_group() -> None:
    assert set(expand_targets("budgies", FINDABLE)) == {"bambi", "budgie"}


def test_multiple_birds_union() -> None:
    assert expand_targets("percy and matcha", FINDABLE) == ["percy", "matcha"]


def test_birds_and_any_mean_everything() -> None:
    every = sorted(b for b in FINDABLE if b != "unknown_bird")
    assert expand_targets("birds", FINDABLE) == every
    assert expand_targets("any bird", FINDABLE) == every
    assert expand_targets("all the birds", FINDABLE) == every


def test_unknown_target_resolves_empty() -> None:
    assert expand_targets("dog", FINDABLE) == []
    assert expand_targets("", FINDABLE) == []


def test_default_species_members_cover_the_flock() -> None:
    assert DEFAULT_SPECIES_MEMBERS["cockatiel"] == ("draft", "pizza")
    assert "percy" in DEFAULT_SPECIES_MEMBERS["lovebird"]


def test_pronoun_map_matches_known_sexes() -> None:
    pronouns = pronoun_map(load_sexes())
    assert pronouns["percy"] == "she"
    assert pronouns["bambi"] == "she"
    assert pronouns["matcha"] == "he"
    assert pronouns["pizza"] == "he"
    assert pronouns["jynx"] == "he"
    assert pronouns["draft"] == "he"


def test_pronoun_sentence_groups_by_sex() -> None:
    note = pronoun_sentence(pronoun_map(load_sexes()))
    assert "Bambi" in note and "Percy" in note and "female (she/her)" in note
    assert "Draft" in note and "male (he/him)" in note
    # Females listed under she, not he.
    she_part = note.split("female")[0]
    assert "Percy" in she_part and "Bambi" in she_part
