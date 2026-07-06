from __future__ import annotations

import json
from datetime import datetime, timezone

from lib.activity import (
    _ACTIVITY_QA_PROMPT,
    _ACTIVITY_SUMMARY_PROMPT,
    _strip_stray_yesno,
    load_sightings,
    select_highlights,
    summarise_counts,
)


def test_strip_stray_yesno_drops_prefix_on_open_questions() -> None:
    # An open "what/who/when" question must not start with a stray Yes/No.
    assert _strip_stray_yesno("Yes. Around sunrise they gathered near the gym.",
                              "what did they do when the sun rose?") == \
        "Around sunrise they gathered near the gym."
    assert _strip_stray_yesno("Yes, Percy was the most active today.",
                              "who was most active?") == "Percy was the most active today."
    assert _strip_stray_yesno("No — they mostly rested this morning.",
                              "what did the birds do this morning?") == "They mostly rested this morning."


def test_strip_stray_yesno_keeps_prefix_on_yesno_questions() -> None:
    # A genuine yes/no question legitimately opens with Yes/No — leave it.
    assert _strip_stray_yesno("Yes, Percy and Matcha were together around 2pm.",
                              "was percy with matcha today?").startswith("Yes")
    assert _strip_stray_yesno("No, I didn't catch Draft flying today.",
                              "did draft fly today?").startswith("No")


def _write_sighting(collect_dir, label, conf, when_epoch, camera="camera-192.168.1.8") -> None:
    folder = collect_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{int(when_epoch)}_{int(conf*100)}"
    (folder / f"{stem}.jpg").write_bytes(b"\xff\xd8jpeg")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "object": label,
                "camera": {"name": camera},
                "collected_at": datetime.fromtimestamp(when_epoch, timezone.utc).isoformat(),
                "frame": {"width": 2304, "height": 1296},
                "detection": {"confidence": conf, "bbox_xyxy": {"x1": 10, "y1": 10, "x2": 30, "y2": 50}},
            }
        )
    )


def test_load_sightings_filters_by_time_and_skips_snapshots(tmp_path) -> None:
    _write_sighting(tmp_path, "percy", 0.9, 1000)
    _write_sighting(tmp_path, "matcha", 0.7, 500)   # before window
    # The snapshots folder must be ignored.
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "x.json").write_text("{}")

    sightings = load_sightings(tmp_path, since_epoch=900)

    assert [s.label for s in sightings] == ["percy"]
    assert sightings[0].confidence == 0.9
    assert sightings[0].camera == "camera-192.168.1.8"


def test_select_highlights_one_per_bird_then_backfill(tmp_path) -> None:
    _write_sighting(tmp_path, "percy", 0.9, 1000)
    _write_sighting(tmp_path, "percy", 0.6, 1001)
    _write_sighting(tmp_path, "matcha", 0.8, 1002)
    sightings = load_sightings(tmp_path, since_epoch=0)

    one_each = select_highlights(sightings, max_photos=2)
    assert {s.label for s in one_each} == {"percy", "matcha"}  # diverse first

    # With room, the second percy shot backfills.
    all_three = select_highlights(sightings, max_photos=3)
    assert sum(1 for s in all_three if s.label == "percy") == 2


def test_summarise_counts(tmp_path) -> None:
    _write_sighting(tmp_path, "percy", 0.9, 1000)
    _write_sighting(tmp_path, "percy", 0.8, 1001)
    _write_sighting(tmp_path, "matcha", 0.7, 1002)
    counts = summarise_counts(load_sightings(tmp_path, 0))
    assert counts.startswith("Percy ×2")
    assert "Matcha ×1" in counts


def test_activity_prompts_keep_answers_short() -> None:
    summary = _ACTIVITY_SUMMARY_PROMPT.lower()
    qa = _ACTIVITY_QA_PROMPT.lower()
    assert "2-4 bullet" in summary
    assert "under 16 words" in summary
    assert "2-3 short sentences" in qa
    assert "counts" in qa  # it must lean on the exact count tallies
    # gemma3 was emitting **markdown headers** — both prompts now forbid it.
    assert "no markdown" in summary and "no markdown" in qa
