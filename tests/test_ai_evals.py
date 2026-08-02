"""Unit tests for the model eval suite's pure parts (no model calls)."""

from __future__ import annotations

import json
from pathlib import Path

from lib.ai.evals import checks, vlm_eval
from lib.ai.evals.runner import _percentiles, REQUIREMENTS, ROLE_TASKS


def test_species_word_detection():
    assert checks.contains_species_word("Percy the lovebird is out")
    assert checks.contains_species_word("The cockatoo perches calmly")
    assert not checks.contains_species_word("Percy is out on her perch with the other birds")


def test_overlay_word_detection():
    assert checks.contains_overlay_word("the bird in the green box")
    assert not checks.contains_overlay_word("Percy preens on a branch")


def test_vocative_opening():
    assert checks.opens_with_vocative_bird_name("Hello, Percy! It's a lovely day.")
    assert checks.opens_with_vocative_bird_name("Percy, it's daytime.")
    assert not checks.opens_with_vocative_bird_name("Percy is preening on the perch.")
    assert not checks.opens_with_vocative_bird_name("Good morning! All four cameras are live.")


def test_invented_numbers():
    facts = "dark 11h 41m (18:41-06:22), score 86/100"
    assert checks.invented_numbers("They slept 11h 41m, scoring 86.", facts) == set()
    assert "9" in checks.invented_numbers("They got 9 hours of sleep.", facts)


def test_yes_no_detection():
    assert checks.starts_yes("Yes — they preened together at 9.")
    assert checks.starts_no("No, they were never seen together.")
    assert not checks.starts_yes_or_no("Jynx was the most active today.")
    # "Nope" counts as a No.
    assert checks.starts_no("Nope, nothing like that.")


def test_bullet_summary_contract():
    good = "• Pizza cracked seeds at the bowl this morning.\n• He chewed the rope toy after lunch."
    assert checks.valid_bullet_summary(good) == []
    assert checks.valid_bullet_summary("Pizza had a fine day.")  # no bullets, 1 line
    assert checks.valid_bullet_summary("• one\n• two\n• three\n• four\n• five")  # 5 lines


def test_sentence_count():
    assert checks.sentence_count("One sentence only") == 1
    assert checks.sentence_count("First. Second! Third?") == 3


def test_meta_word_detection():
    assert checks.contains_meta_word("Based on the counts, Percy rested.")
    assert not checks.contains_meta_word("Percy rested most of the day.")


def test_percentiles():
    p50, p90 = _percentiles([1.0, 2.0, 3.0, 4.0, 10.0])
    assert p50 == 3.0
    assert p90 == 10.0
    assert _percentiles([]) == (0.0, 0.0)


def test_requirements_cover_all_role_tasks():
    for tasks in ROLE_TASKS.values():
        for task in tasks:
            assert task in REQUIREMENTS


def test_activity_classes_cover_vocabulary():
    from lib.ai.vlm import BIRD_ACTIVITIES

    for activity in BIRD_ACTIVITIES:
        assert activity in vlm_eval.ACTIVITY_CLASSES


def test_sample_observations_deterministic(tmp_path, monkeypatch):
    # Two runs over the same journal pick the same frames in the same order.
    images = tmp_path / "images"
    images.mkdir()
    day = tmp_path / "2026-07-01.jsonl"
    entries = []
    for idx in range(6):
        photo = images / f"frame{idx}.jpg"
        photo.write_bytes(b"\xff\xd8fake\xff\xd9")
        entries.append(json.dumps({
            "version": 3,
            "time": f"2026-07-01T0{idx}:00:00",
            "birds": ["percy"],
            "note": "",
            "photos": [str(photo)],
            "observations": [{
                "camera": f"Cam {idx % 2}",
                "birds": ["percy"],
                "note": "Percy preens.",
                "photo": str(photo),
                "detections": [{
                    "label": "percy", "confidence": 0.9,
                    "bbox": [10, 10, 100, 100],
                    "activity": "preening" if idx % 2 else "resting",
                }],
            }],
        }))
    day.write_text("\n".join(entries), encoding="utf-8")
    first = vlm_eval.sample_observations(tmp_path, limit=4)
    second = vlm_eval.sample_observations(tmp_path, limit=4)
    assert [str(s.photo) for s in first] == [str(s.photo) for s in second]
    assert first and first[0].silver
