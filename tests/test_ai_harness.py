from __future__ import annotations

from lib.ai.harness import INTENT_TESTS, QUICK_TESTS, _capabilities
from lib.ai.intent import ACTIONS


def test_intent_test_cases_use_valid_actions() -> None:
    for message, expected in INTENT_TESTS:
        assert expected in ACTIONS, f"{message!r} expects unknown action {expected!r}"


def test_quick_tests_are_a_subset() -> None:
    assert set(QUICK_TESTS).issubset(set(INTENT_TESTS))
    assert 0 < len(QUICK_TESTS) < len(INTENT_TESTS)


def test_capabilities_reads_either_shape() -> None:
    assert _capabilities({"capabilities": ["vision"]}) == ["vision"]
    assert _capabilities({"details": {"capabilities": ["completion"]}}) == ["completion"]
    assert _capabilities({}) == []
