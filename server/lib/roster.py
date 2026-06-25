"""Expand a free-text find target into concrete detector labels.

``/find`` (and "find the cockatiels") needs to turn what a person says into the
set of labels the live model can actually detect. A request can name an
individual ("percy"), a species/group ("cockatiels", "lovebirds"), several birds
("percy and matcha"), or everything ("the birds"). This module resolves all of
those against the model's real label set, so a group like "cockatiels" becomes
its individuals (draft, pizza) plus the IR species outline ("cockatiel").

The species→members map is loaded best-effort from the training roster
(``training/roster.yaml``); if that file isn't deployed (e.g. a slim server
image) it falls back to a built-in map of the current flock.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path


LOGGER = logging.getLogger("lib.roster")

# Fallback species → individual members, used when roster.yaml isn't available.
DEFAULT_SPECIES_MEMBERS: dict[str, tuple[str, ...]] = {
    "cockatiel": ("draft", "pizza"),
    "lovebird": ("percy", "matcha", "jynx"),
    "budgie": ("bambi",),
}

# Bird sexes, so descriptions use the right pronoun. Loaded from roster.yaml's
# ``sex`` field when present; this is the fallback for the current flock.
DEFAULT_SEXES: dict[str, str] = {
    "jynx": "male", "matcha": "male", "draft": "male", "pizza": "male",
    "percy": "female", "bambi": "female",
}

_PRONOUNS = {"male": "he", "female": "she"}

# Words that mean "every bird". "any"/"anyone" land here too: "find any bird"
# resolves to all birds, and the search reports the FIRST one it sees.
ALL_BIRD_WORDS = {
    "bird", "everyone", "everybody", "all", "any", "anyone", "anybody", "everything",
}

# Filler dropped during tokenisation so "find the cockatiels" -> ["cockatiels"].
# Note "all"/"any" are deliberately NOT here — they are bird words above.
_STOPWORDS = {
    "the", "a", "an", "find", "locate", "where", "is", "are", "s",
    "my", "our", "please", "for", "me", "to", "and", "or", "both",
}

# Default candidate roster paths, tried in order (cwd-relative first).
_DEFAULT_ROSTER_PATHS = (
    Path("training/roster.yaml"),
    Path(__file__).resolve().parents[2] / "training" / "roster.yaml",
)


def load_species_members(
    paths: tuple[Path, ...] = _DEFAULT_ROSTER_PATHS,
) -> dict[str, tuple[str, ...]]:
    """Build species → individual-member map from roster.yaml, or the fallback.

    Reads each label's ``species`` field and groups the individuals under it.
    Any read/parse problem (missing file, bad YAML) falls back to
    :data:`DEFAULT_SPECIES_MEMBERS` so the feature never hard-fails on a slim
    deployment that ships without the training tree.
    """
    for path in paths:
        try:
            if not path.exists():
                continue
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            members: dict[str, list[str]] = {}
            for entry in data.get("labels", []):
                if entry.get("kind") != "individual":
                    continue
                species = entry.get("species")
                name = entry.get("name")
                if species and name:
                    members.setdefault(str(species).lower(), []).append(str(name).lower())
            if members:
                return {species: tuple(names) for species, names in members.items()}
        except Exception:
            LOGGER.exception("Failed to load roster from %s; using fallback", path)
    return DEFAULT_SPECIES_MEMBERS


def _singularise(word: str) -> str:
    """Crude depluraliser: cockatiels->cockatiel, budgies->budgie, birds->bird.

    Just drops a trailing "s" — every plural we care about (the species and
    "birds") is formed that way. We avoid the "ies"->"y" rule on purpose: it
    would wrongly turn "budgies" into "budgy" rather than "budgie".
    """
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def expand_targets(
    text: str,
    findable_labels: list[str],
    species_members: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Resolve free-text ``text`` into a de-duplicated list of findable labels.

    - an individual name -> itself ("percy" -> [percy])
    - a species/group -> its members + the species outline label, intersected
      with what's findable ("cockatiels" -> [draft, pizza, cockatiel])
    - "birds"/"everyone"/"all" -> every findable bird (minus the unknown class)
    - several at once -> the union ("percy and matcha" -> [percy, matcha])

    Anything not recognised is ignored; an empty result means "nothing findable
    matched", which the caller surfaces as a helpful error.
    """
    species_members = species_members or DEFAULT_SPECIES_MEMBERS
    findable = {label.lower() for label in findable_labels}
    every_bird = sorted(label for label in findable if label != "unknown_bird")

    tokens = [
        token
        for token in re.split(r"[\s,&/]+", text.lower().strip())
        if token and token.strip(".!?'\"") not in _STOPWORDS
    ]

    result: list[str] = []
    for raw in tokens:
        token = raw.strip(".!?'\"")
        if not token:
            continue
        singular = _singularise(token)
        if token in ALL_BIRD_WORDS or singular in ALL_BIRD_WORDS:
            result.extend(every_bird)
            continue
        if singular in species_members:
            result.extend(member for member in species_members[singular] if member in findable)
            if singular in findable:  # the IR species outline, e.g. "cockatiel"
                result.append(singular)
            continue
        if token in findable:
            result.append(token)
        elif singular in findable:
            result.append(singular)

    # De-duplicate, preserving first-seen order.
    seen: set[str] = set()
    ordered: list[str] = []
    for label in result:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def load_sexes(paths: tuple[Path, ...] = _DEFAULT_ROSTER_PATHS) -> dict[str, str]:
    """Map each individual bird to its sex from roster.yaml, or the fallback."""
    for path in paths:
        try:
            if not path.exists():
                continue
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            sexes = {
                str(entry["name"]).lower(): str(entry["sex"]).lower()
                for entry in data.get("labels", [])
                if entry.get("kind") == "individual" and entry.get("sex")
            }
            if sexes:
                return sexes
        except Exception:
            LOGGER.exception("Failed to load sexes from %s; using fallback", path)
    return DEFAULT_SEXES


def pronoun_map(sexes: dict[str, str] | None = None) -> dict[str, str]:
    """Build a name -> "he"/"she" map (birds of unknown sex are omitted -> "they")."""
    sexes = sexes if sexes is not None else DEFAULT_SEXES
    return {name: _PRONOUNS[sex] for name, sex in sexes.items() if sex in _PRONOUNS}


def pronoun_sentence(pronouns: dict[str, str]) -> str:
    """A ground-truth sentence for prompts: "Percy and Bambi are female (she/her); ...".

    The captions/observations fed to the summariser may not carry pronouns, so
    the model defaults birds to "he". Stating the sexes explicitly fixes that.
    """
    from lib.labels import pretty

    she = sorted(name for name, p in pronouns.items() if p == "she")
    he = sorted(name for name, p in pronouns.items() if p == "he")

    def _clause(names: list[str], word: str, pronoun: str) -> str:
        joined = ", ".join(pretty(n) for n in names)
        verb = "is" if len(names) == 1 else "are"
        return f"{joined} {verb} {word} (use {pronoun} for them)"

    parts = []
    if she:
        parts.append(_clause(she, "female", "she/her"))
    if he:
        parts.append(_clause(he, "male", "he/him"))
    return ". ".join(parts) + "." if parts else ""
