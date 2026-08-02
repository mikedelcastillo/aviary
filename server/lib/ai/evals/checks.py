"""Pure text checks shared by the eval tasks (no model, fully unit-testable).

Each check encodes one rule from a production prompt contract — the chat
persona's "never a species word", the QA prompt's "no meta-words", the summary
prompt's bullet format — so a candidate model is graded against exactly what
the live prompts demand, not a vibe.
"""

from __future__ import annotations

import re


# Species/breed words the personas ban ("never 'Percy the lovebird'"). "bird"
# itself is allowed. Includes the common wrong-species words models substitute
# (the 7b incumbent has been seen writing "cockatoo" for a cockatiel).
SPECIES_WORDS = {
    "lovebird", "lovebirds", "cockatiel", "cockatiels", "budgie", "budgies",
    "budgerigar", "budgerigars", "parakeet", "parakeets", "parrot", "parrots",
    "cockatoo", "cockatoos", "finch", "finches", "canary", "canaries",
    "conure", "conures", "macaw", "macaws",
}

# Words that reveal the VLM is describing the annotation overlay, not the birds
# ("the bird in the green box") — banned by ANALYZE_PROMPT.
OVERLAY_WORDS = {
    "box", "boxes", "boxed", "outline", "outlined", "rectangle", "rectangles",
    "label", "labels", "labeled", "labelled", "overlay", "bounding", "marker",
    "annotation", "annotated", "highlighted",
}

# Meta-words the recall QA prompt forbids quoting back at the user.
RECALL_META_WORDS = {"counts", "notes", "window", "observations", "tallies", "data"}

BIRD_NAMES = ("percy", "matcha", "jynx", "bambi", "draft", "pizza")


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    """Rough sentence count — enough to catch a 5-sentence essay vs 'one short
    sentence'. Trailing fragments without terminal punctuation count as one."""
    parts = [p for p in re.split(r"[.!?]+(?:\s|$)", text.strip()) if p.strip()]
    return max(1, len(parts))


def contains_species_word(text: str) -> bool:
    return bool(set(words_of(text)) & SPECIES_WORDS)


def contains_overlay_word(text: str) -> bool:
    return bool(set(words_of(text)) & OVERLAY_WORDS)


def contains_meta_word(text: str) -> bool:
    return bool(set(words_of(text)) & RECALL_META_WORDS)


def mentions_any(text: str, names) -> bool:
    tokens = set(words_of(text))
    return any(str(n).lower() in tokens for n in names)


def mentions_all(text: str, names) -> bool:
    tokens = set(words_of(text))
    return all(str(n).lower() in tokens for n in names)


def opens_with_vocative_bird_name(text: str) -> bool:
    """Catches "Hello, Percy!" / "Percy, it's daytime" — the chat prompt bans
    greeting the BIRD; a bird's name may only be the subject of a statement.

    "Percy, Matcha, and Jynx are roosting" is an ENUMERATION, not a vocative —
    a leading name followed by another bird name (or "and") stays allowed.
    """
    s = text.strip()
    m = re.match(r"^(?:hi|hello|hey|good\s+\w+|goodnight)[,!\s]+([a-z]+)", s, re.IGNORECASE)
    if m and m.group(1).lower() in BIRD_NAMES:
        return True
    m = re.match(rf"^({'|'.join(BIRD_NAMES)}),\s+(\w+)", s, re.IGNORECASE)
    if not m:
        return False
    follower = m.group(2).lower()
    return follower not in BIRD_NAMES and follower != "and"


def has_markdown(text: str) -> bool:
    """Bold/headers/tables — banned in every plain-text Telegram contract."""
    return bool(
        re.search(r"\*\*|__|^#+\s|^\s*\|.*\|", text, re.MULTILINE)
    )


def numbers_in(text: str) -> set[str]:
    """Every number token, normalised (times split into components, '86/100'
    into both halves) — used to assert a reply invents no figure that isn't in
    its source facts."""
    return set(re.findall(r"\d+", text))


def invented_numbers(reply: str, source: str) -> set[str]:
    """Numbers present in ``reply`` but nowhere in ``source``.

    Small allowances: a count the model spells out is not caught (fine — the
    check targets confabulated tallies/times, which models write as digits).
    """
    return numbers_in(reply) - numbers_in(source)


def starts_yes(text: str) -> bool:
    return bool(re.match(r"^\s*yes\b", text.strip(), re.IGNORECASE))


def starts_no(text: str) -> bool:
    return bool(re.match(r"^\s*no(?:pe)?\b", text.strip(), re.IGNORECASE))


def starts_yes_or_no(text: str) -> bool:
    return starts_yes(text) or starts_no(text)


def bullet_lines(text: str) -> list[str]:
    return [line for line in text.strip().splitlines() if line.strip()]


def valid_bullet_summary(text: str) -> list[str]:
    """Violations of the activity-summary contract (2-4 '• ' lines, <16 words
    each, nothing else). Empty list = compliant."""
    problems: list[str] = []
    lines = bullet_lines(text)
    if not 2 <= len(lines) <= 4:
        problems.append(f"{len(lines)} lines (want 2-4)")
    for line in lines:
        if not line.strip().startswith("• ") and not line.strip().startswith("- "):
            problems.append(f"line missing bullet: {line[:40]!r}")
        # The prompt says <16 words; allow a little slack before failing so a
        # 17-word line doesn't sink an otherwise perfect summary.
        if len(line.split()) > 20:
            problems.append(f"line too long ({len(line.split())} words)")
    if has_markdown(text):
        problems.append("markdown present")
    return problems
