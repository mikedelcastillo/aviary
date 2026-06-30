"""Shared, user-facing formatting for bird labels and confidences.

Bird names are proper nouns, so everything the user sees — alert captions,
snapshot boxes, /status, /find — Capitalises them, and confidences are shown as
whole percentages (85%) rather than decimals (0.85). Centralised here so every
surface formats identically.
"""

from __future__ import annotations

from collections.abc import Iterable


def pretty(label: str) -> str:
    """Capitalise a label for display: percy -> Percy, unknown_bird -> Unknown Bird."""
    return label.replace("_", " ").title()


def pretty_labels(labels: Iterable[str]) -> str:
    """De-duplicate, sort and Capitalise a set of labels into "Bambi, Percy"."""
    return ", ".join(pretty(label) for label in sorted(set(labels)))


def format_confidence(confidence: float) -> str:
    """Render a 0..1 confidence as a whole percentage: 0.85 -> "85%"."""
    return f"{round(confidence * 100)}%"
