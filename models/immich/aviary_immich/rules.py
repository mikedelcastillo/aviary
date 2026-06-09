"""Album rules: how model signals combine into album membership.

A record carries per-model, per-tag confidences under ``record["model_tags"]`` (provenance written
by :func:`aviary_immich.records.record_from_outputs`). An :class:`AlbumRule` decides whether that
provenance puts the asset in an album:

- ``union`` (default): the album fires if ANY of its signals matched — maximizes recall.
- ``agreement``: the album fires only when at least ``min_votes`` DISTINCT models concur on a
  signal — maximizes precision (the "second opinion").

Tennis is a union of two different signals from two different models (``yolo:tennis racket`` OR
``clip:tennis court``); Birds can become an agreement of ``yolo:bird`` AND ``clip:bird``.

These functions are pure (plain dicts in, sets/floats out) and import nothing heavy, so they are
directly unit-testable and safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Signal:
    """One model emitting one tag, optionally with its own minimum confidence.

    ``model`` is a provenance key (e.g. ``"yolo"``, ``"clip"``) or ``"*"`` to accept the tag from
    any model. ``min_confidence=None`` accepts whatever confidence the model already thresholded at.
    """

    model: str
    tag: str
    min_confidence: float | None = None


@dataclass(frozen=True)
class AlbumRule:
    """Maps a set of signals to one album under a combination ``mode``."""

    album_name: str
    signals: tuple[Signal, ...]
    mode: str = "union"  # "union" | "agreement"
    min_votes: int = 1   # agreement: how many DISTINCT models must concur


def provenance(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return ``{model: {tag: max_confidence}}`` for a record.

    Prefers the explicit ``model_tags`` field. Falls back, for records written before multi-model
    support, to synthesizing provenance from per-detection ``model``/``label`` (or the flat
    ``labels`` field under the legacy ``"yolo"`` key) so old state JSONL still routes.
    """
    raw = record.get("model_tags")
    if raw:
        return {
            str(model): {str(tag).lower(): float(conf) for tag, conf in tags.items()}
            for model, tags in raw.items()
        }

    synthesized: dict[str, dict[str, float]] = {}
    for detection in record.get("detections", []) or []:
        name = str(detection.get("label", "")).lower()
        if not name:
            continue
        model = str(detection.get("model") or "yolo")
        conf = float(detection.get("confidence") or 0)
        bucket = synthesized.setdefault(model, {})
        bucket[name] = max(bucket.get(name, 0.0), conf)
    if synthesized:
        return synthesized

    labels = record.get("labels")
    if labels:
        conf = float(record.get("max_confidence") or 0)
        return {"yolo": {str(label).lower(): conf for label in labels if label}}
    return {}


def _voters(prov: dict[str, dict[str, float]], signal: Signal) -> set[str]:
    """Distinct model names that satisfy ``signal`` given the provenance."""
    models = prov.keys() if signal.model == "*" else ([signal.model] if signal.model in prov else [])
    voters: set[str] = set()
    for model in models:
        conf = prov.get(model, {}).get(signal.tag)
        if conf is None:
            continue
        if signal.min_confidence is not None and conf < signal.min_confidence:
            continue
        voters.add(model)
    return voters


def evaluate_rules(record: dict[str, Any], rules: tuple[AlbumRule, ...]) -> set[str]:
    """Return the set of album names a record matches."""
    prov = provenance(record)
    matched: set[str] = set()
    for rule in rules:
        voters: set[str] = set()
        for signal in rule.signals:
            voters |= _voters(prov, signal)
        if rule.mode == "agreement":
            if len(voters) >= rule.min_votes:
                matched.add(rule.album_name)
        elif voters:  # union (default)
            matched.add(rule.album_name)
    return matched


def rule_confidence(record: dict[str, Any], rule: AlbumRule) -> float:
    """Best confidence among the rule's signals that fired (for the manifest column)."""
    prov = provenance(record)
    best = 0.0
    for signal in rule.signals:
        models = prov.keys() if signal.model == "*" else [signal.model]
        for model in models:
            conf = prov.get(model, {}).get(signal.tag)
            if conf is not None:
                best = max(best, conf)
    return best
