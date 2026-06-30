from __future__ import annotations

import json
from datetime import date

from lib.journal import load_entries
from lib.memory_cleanup import migrate_memories


def test_cleanup_splits_legacy_camera_observations(tmp_path) -> None:
    (tmp_path / "2026-06-25.md").write_text(
        "# Aviary memories — 2026-06-25\n\n"
        "## 09:00 | percy, matcha\n"
        "Percy on Big Cage: Percy preened on a perch.; "
        "Matcha on Desk: Matcha chewed a toy.\n"
        "> photo: p.jpg\n"
        "> photo: m.jpg\n\n",
        encoding="utf-8",
    )

    stats = migrate_memories(tmp_path, backup=False)

    assert stats.entries == 1
    assert stats.observations == 2
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert entries[0].birds == ["matcha", "percy"]
    assert entries[0].observations[0].camera == "Big Cage"
    assert entries[0].observations[0].birds == ["percy"]
    assert entries[0].observations[0].photo == "p.jpg"
    text = (tmp_path / "2026-06-25.md").read_text(encoding="utf-8")
    assert "- Percy on Big Cage: Percy preened on a perch." in text
    records = [
        json.loads(line)
        for line in (tmp_path / "2026-06-25.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["observations"][1]["birds"] == ["matcha"]


def test_cleanup_does_not_count_uncertain_legacy_captions_as_bird_activity(tmp_path) -> None:
    (tmp_path / "2026-06-25.md").write_text(
        "# Aviary memories — 2026-06-25\n\n"
        "## 09:00 | percy\n"
        "Percy on Big Cage: The cage appears empty with no birds visible.\n"
        "> photo: p.jpg\n\n",
        encoding="utf-8",
    )

    stats = migrate_memories(tmp_path, backup=False)

    assert stats.uncertain_observations == 1
    assert stats.bird_labels_removed == 1
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert entries[0].birds == []
    assert entries[0].observations[0].birds == []
    assert "## 09:00 | quiet" in (tmp_path / "2026-06-25.md").read_text(encoding="utf-8")


def test_cleanup_does_not_treat_possessive_names_as_presence(tmp_path) -> None:
    (tmp_path / "2026-06-25.md").write_text(
        "# Aviary memories — 2026-06-25\n\n"
        "## 09:00 | cockatiel\n"
        "• Bambi's cockatiel perched on her cage branch.\n"
        "> photo: c.jpg\n\n",
        encoding="utf-8",
    )

    migrate_memories(tmp_path, backup=False)

    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert entries[0].birds == ["cockatiel"]
    assert entries[0].observations[0].birds == ["cockatiel"]
