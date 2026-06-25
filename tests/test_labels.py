from __future__ import annotations

from lib.labels import format_confidence, pretty, pretty_labels


def test_pretty_capitalises_and_expands_underscores() -> None:
    assert pretty("percy") == "Percy"
    assert pretty("unknown_bird") == "Unknown Bird"


def test_pretty_labels_dedupes_sorts_and_capitalises() -> None:
    assert pretty_labels(["percy", "bambi", "percy"]) == "Bambi, Percy"


def test_format_confidence_is_whole_percent() -> None:
    assert format_confidence(0.85) == "85%"
    assert format_confidence(0.5) == "50%"
    assert format_confidence(0.004) == "0%"
    assert format_confidence(1.0) == "100%"
