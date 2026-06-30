from __future__ import annotations

from pathlib import Path

from lib.ai.activity_harness import (
    QA_CASES,
    QA_DAY,
    SYNTHETIC_ENTRIES,
    build_journal,
    score,
)
from lib.ai.intent import ACTIONS
from lib.journal import load_entries


def test_score_flags_missing_groups() -> None:
    # All groups satisfied -> nothing missing.
    assert score("Yes, Pizza ate seeds this morning", [["pizza"], ["yes", "ate"]]) == []
    # A missing group is reported.
    missing = score("Pizza perched quietly", [["pizza"], ["ate", "eat", "food"]])
    assert missing == ["ate / eat / food"]


def test_score_is_case_insensitive() -> None:
    assert score("BAMBI had a BATH", [["bambi"], ["bath"]]) == []


def test_qa_cases_route_to_known_actions() -> None:
    for question, action, groups in QA_CASES:
        assert action in ACTIONS
        assert groups and all(isinstance(g, list) and g for g in groups)


def test_build_journal_is_readable(tmp_path: Path) -> None:
    build_journal(tmp_path)
    entries = load_entries(tmp_path, QA_DAY)
    assert len(entries) == len(SYNTHETIC_ENTRIES)
    # The co-occurrence entry keeps both birds, for "together" questions.
    together = [e for e in entries if set(e.birds) == {"jynx", "matcha"}]
    assert len(together) == 1
