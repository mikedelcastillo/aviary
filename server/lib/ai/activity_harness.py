"""End-to-end harness for memory lookback + VLM activity tracking.

``uv run activity-harness`` checks two things the day-Q&A relies on:

  * VLM detail — runs the vision model on real collected frames (with the same
    detection grounding the server uses) and checks each caption names a concrete
    ACTIVITY (eating, preening, bathing, …), not just "a bird". Reports coverage.

  * Memory lookback — builds a synthetic day journal with known events, then runs
    real day-questions ("did pizza eat today?", "did jynx and matcha spend time
    together?", "when did draft go in the cage?", and a negative "did pizza take a
    bath?") through the FULL pipeline (intent routing -> ActivityResponder -> LLM
    answer) and scores each answer against expected facts.

It talks to the same Ollama/Olla the server uses (OLLAMA_BASE_URL).
"""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
import time
from datetime import date, datetime, time as dtime
from pathlib import Path

from lib.activity import _BoxDetection, _parse_sidecar
from lib.activity_qa import ActivityResponder
from lib.ai.client import OllamaClient
from lib.ai.intent import classify_intent
from lib.ai.vlm import build_detection_context, describe_image
from lib.config import _ollama_config
from lib.roster import load_sexes, load_species_members, pronoun_map, pronoun_sentence


DEFAULT_BIRDS = [
    "bambi", "budgie", "cockatiel", "draft", "jynx",
    "lovebird", "matcha", "percy", "pizza", "unknown_bird",
]

# Concrete activity words a good caption should contain (vs a bare "a bird").
ACTIVITY_WORDS = {
    "eat", "eating", "ate", "feed", "feeding", "seed", "seeds", "food", "bowl",
    "perch", "perched", "perching", "preen", "preening", "groom", "grooming",
    "play", "playing", "toy", "bell", "drink", "drinking", "water", "bath",
    "bathing", "splash", "sleep", "sleeping", "nap", "napping", "rest", "resting",
    "fly", "flying", "climb", "climbing", "hang", "hanging", "sit", "sitting",
    "stand", "standing", "watch", "watching", "looking", "chew", "chewing",
    "cage", "branch", "wing", "wings",
}

# Synthetic day (PH wall-clock). Each: (time, birds, note).
QA_DAY = date(2026, 6, 25)
QA_NOW = datetime(2026, 6, 25, 17, 0)
SYNTHETIC_ENTRIES: list[tuple[dtime, list[str], str]] = [
    (dtime(7, 30), ["pizza"], "Pizza cracked seeds at the food bowl, eating hungrily."),
    (dtime(9, 0), ["jynx", "matcha"], "Jynx and Matcha preened side by side on the window perch."),
    (dtime(10, 15), ["bambi"], "Bambi splashed around in the water dish, having a proper bath."),
    (dtime(12, 30), ["draft"], "Draft climbed back into the cage and settled on a branch."),
    (dtime(14, 0), ["percy"], "Percy napped quietly with her eyes closed, tucked on the perch."),
    (dtime(16, 30), ["jynx"], "Jynx played with the little bell toy."),
]

# (question, expected_action, [groups]) — each group is a list of acceptable
# synonyms; the answer must contain at least one token from EVERY group.
QA_CASES: list[tuple[str, str, list[list[str]]]] = [
    ("did pizza eat today?", "activity",
     [["pizza"], ["yes", "ate", "eat", "eating", "seed", "food"]]),
    ("did jynx and matcha spend time together today?", "activity",
     [["jynx"], ["matcha"], ["yes", "together", "side by side", "both"],
      ["9", "morning", "nine", "ago", "earlier"]]),
    ("did the birds take a bath today?", "activity",
     [["bambi"], ["yes", "bath", "splash", "water"]]),
    ("when did draft go in the cage?", "activity",
     [["draft"], ["12", "noon", "afternoon", "half past", "cage", "ago", "midday"]]),
    ("what did percy do today?", "activity",
     [["percy"], ["nap", "slept", "sleep", "rest", "quiet", "doze"]]),
    # Negative: pizza did NOT bathe (bambi did) — must not hallucinate a yes.
    ("did pizza take a bath today?", "activity",
     [["no", "didn't", "did not", "not", "no sign", "only bambi", "wasn't"]]),
]


def build_journal(memories_dir: Path) -> None:
    from lib.journal import MemoryEntry, append_entry

    for when, birds, note in SYNTHETIC_ENTRIES:
        append_entry(memories_dir, MemoryEntry(datetime.combine(QA_DAY, when), birds, note))


def score(answer: str, groups: list[list[str]]) -> list[str]:
    """Return the list of expected groups NOT satisfied by ``answer``."""
    low = answer.lower()
    return [
        " / ".join(group)
        for group in groups
        if not any(token.lower() in low for token in group)
    ]


# -- memory lookback --------------------------------------------------------


def run_qa(client: OllamaClient, cfg, birds: list[str]) -> tuple[int, int]:
    print("\n\033[1m=== memory lookback Q&A ===\033[0m")
    pronoun_note = pronoun_sentence(pronoun_map(load_sexes()))
    passed = 0
    with tempfile.TemporaryDirectory() as tmp:
        memories_dir = Path(tmp)
        build_journal(memories_dir)
        for question, expect_action, groups in QA_CASES:
            captured: list[str] = []
            responder = ActivityResponder(
                memories_dir, client, cfg.llm_model, lambda: birds,
                notify=lambda cid, text: captured.append(text),
                send_album=None,
                pronoun_note=pronoun_note,
                now=lambda: QA_NOW,
            )
            t0 = time.time()
            try:
                intent = classify_intent(client, cfg.llm_model, question, birds)
                routed_ok = intent.action == expect_action
                # Even if routing missed, still answer with the named argument so
                # we can see the answer quality.
                responder.respond(0, question, intent.argument or "")
            except Exception as exc:
                print(f"  ✗ {question!r}: ERROR {exc}")
                continue
            answer = captured[-1] if captured else ""
            missing = score(answer, groups)
            ok = routed_ok and not missing
            passed += ok
            mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
            print(f"  {mark} ({time.time()-t0:4.1f}s) {question!r}")
            if not routed_ok:
                print(f"      routed to {intent.action!r}, expected {expect_action!r}")
            if missing:
                print(f"      missing: {', '.join(missing)}")
            print(f"      → {answer[:160]}")
    print(f"\n  \033[1mQ&A score: {passed}/{len(QA_CASES)}\033[0m")
    return passed, len(QA_CASES)


# -- VLM activity detail ----------------------------------------------------


def _sample_sightings(limit: int):
    """A few real collected frames across distinct birds (with sidecars)."""
    sightings = []
    seen_birds: set[str] = set()
    for json_path in sorted(glob.glob("data/server/collect/**/*.json", recursive=True), reverse=True):
        sighting = _parse_sidecar(Path(json_path))
        if sighting is None or not sighting.width:
            continue
        if sighting.label in seen_birds:
            continue
        seen_birds.add(sighting.label)
        sightings.append(sighting)
        if len(sightings) >= limit:
            break
    return sightings


def run_vlm(client: OllamaClient, cfg, limit: int) -> tuple[int, int]:
    print("\n\033[1m=== VLM activity detail ===\033[0m")
    pronouns = pronoun_map(load_sexes())
    sightings = _sample_sightings(limit)
    if not sightings:
        print("  no collected frames under data/server/collect — skip")
        return 0, 0
    detailed = 0
    for s in sightings:
        context = build_detection_context(
            [_BoxDetection(s.label, s.bbox)], s.width, s.height, None, pronouns
        )
        t0 = time.time()
        try:
            caption = describe_image(
                client, cfg.vlm_model, s.path.read_bytes(),
                "In one short sentence, say what this bird is doing.",
                context=context, timeout_seconds=120,
            )
        except Exception as exc:
            print(f"  ✗ {s.label}: ERROR {exc}")
            continue
        has_activity = any(word in caption.lower() for word in ACTIVITY_WORDS)
        detailed += has_activity
        mark = "\033[32m✓\033[0m" if has_activity else "\033[33m?\033[0m"
        print(f"  {mark} ({time.time()-t0:4.1f}s) [{s.label}] {caption[:140]}")
    print(f"\n  \033[1mVLM detail: {detailed}/{len(sightings)} named a concrete activity\033[0m")
    return detailed, len(sightings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aviary memory-lookback + VLM-detail harness")
    parser.add_argument("--qa-only", action="store_true", help="only run the memory Q&A part")
    parser.add_argument("--vlm-only", action="store_true", help="only run the VLM detail part")
    parser.add_argument("--limit", type=int, default=5, help="VLM frames to sample")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    cfg = _ollama_config()
    client = OllamaClient(cfg.base_url, timeout_seconds=cfg.timeout_seconds)
    birds = DEFAULT_BIRDS
    print(f"Ollama: {cfg.base_url}  llm={cfg.llm_model}  vlm={cfg.vlm_model}")
    if not client.is_available():
        print("\033[31mOllama is NOT reachable — is it running? (OLLAMA_BASE_URL)\033[0m")
        sys.exit(1)

    qa = vlm = (0, 0)
    if not args.vlm_only:
        qa = run_qa(client, cfg, birds)
    if not args.qa_only:
        vlm = run_vlm(client, cfg, args.limit)

    # Non-zero exit if the Q&A part regressed (the deterministic half).
    if not args.vlm_only and qa[1] and qa[0] < qa[1]:
        sys.exit(1)


if __name__ == "__main__":
    main()
