"""Unit tests for ``aviary_immich.rules``: how model signals combine into albums.

These functions are pure (plain dicts in, sets/floats out) and import nothing heavy, so the whole
file runs on a plain CPU box with no torch/cv2/ultralytics.
"""

from __future__ import annotations

import fakes

from aviary_immich.rules import (
    AlbumRule,
    Signal,
    evaluate_rules,
    provenance,
    rule_confidence,
)


# --------------------------------------------------------------------------- union


TENNIS = AlbumRule(
    "Tennis",
    (Signal("yolo", "tennis racket"), Signal("clip", "tennis court")),
    mode="union",
)


def test_union_fires_on_first_signal_alone():
    record = {"model_tags": {"yolo": {"tennis racket": 0.9}}}
    assert evaluate_rules(record, (TENNIS,)) == {"Tennis"}


def test_union_fires_on_second_signal_alone():
    record = {"model_tags": {"clip": {"tennis court": 0.8}}}
    assert evaluate_rules(record, (TENNIS,)) == {"Tennis"}


def test_union_fires_when_both_signals_present():
    record = {"model_tags": {"yolo": {"tennis racket": 0.9}, "clip": {"tennis court": 0.8}}}
    assert evaluate_rules(record, (TENNIS,)) == {"Tennis"}


def test_union_does_not_fire_with_no_matching_signal():
    record = {"model_tags": {"yolo": {"dog": 0.9}}}
    assert evaluate_rules(record, (TENNIS,)) == set()


# --------------------------------------------------------------------------- agreement


BIRDS_AGREEMENT = AlbumRule(
    "Birds",
    (Signal("yolo", "bird"), Signal("clip", "bird")),
    mode="agreement",
    min_votes=2,
)


def test_agreement_fires_only_when_both_models_concur():
    record = {"model_tags": {"yolo": {"bird": 0.9}, "clip": {"bird": 0.7}}}
    assert evaluate_rules(record, (BIRDS_AGREEMENT,)) == {"Birds"}


def test_agreement_does_not_fire_for_single_model_yolo():
    record = {"model_tags": {"yolo": {"bird": 0.9}}}
    assert evaluate_rules(record, (BIRDS_AGREEMENT,)) == set()


def test_agreement_does_not_fire_for_single_model_clip():
    record = {"model_tags": {"clip": {"bird": 0.9}}}
    assert evaluate_rules(record, (BIRDS_AGREEMENT,)) == set()


# --------------------------------------------------------------------------- min_confidence


CONFIDENT_BIRD = AlbumRule(
    "Birds",
    (Signal("yolo", "bird", min_confidence=0.5),),
    mode="union",
)


def test_min_confidence_blocks_low_confidence_tag():
    record = {"model_tags": {"yolo": {"bird": 0.3}}}
    assert evaluate_rules(record, (CONFIDENT_BIRD,)) == set()


def test_min_confidence_allows_high_confidence_tag():
    record = {"model_tags": {"yolo": {"bird": 0.7}}}
    assert evaluate_rules(record, (CONFIDENT_BIRD,)) == {"Birds"}


# --------------------------------------------------------------------------- wildcard model


WILDCARD_BIRD = AlbumRule("Birds", (Signal("*", "bird"),), mode="union")


def test_wildcard_matches_tag_from_any_model():
    assert evaluate_rules({"model_tags": {"yolo": {"bird": 0.9}}}, (WILDCARD_BIRD,)) == {"Birds"}
    assert evaluate_rules({"model_tags": {"clip": {"bird": 0.9}}}, (WILDCARD_BIRD,)) == {"Birds"}
    assert evaluate_rules({"model_tags": {"other": {"bird": 0.9}}}, (WILDCARD_BIRD,)) == {"Birds"}


def test_wildcard_does_not_match_absent_tag():
    assert evaluate_rules({"model_tags": {"yolo": {"dog": 0.9}}}, (WILDCARD_BIRD,)) == set()


# --------------------------------------------------------------------------- legacy provenance


def test_legacy_flat_labels_route_under_yolo():
    record = fakes.make_record(labels=["bird"], max_confidence=0.42)
    prov = provenance(record)
    assert prov == {"yolo": {"bird": 0.42}}
    rule = AlbumRule("Birds", (Signal("yolo", "bird"),), mode="union")
    assert evaluate_rules(record, (rule,)) == {"Birds"}


def test_legacy_detections_synthesize_per_model_provenance():
    record = {"detections": [{"label": "dog", "confidence": 0.8, "model": "clip"}]}
    assert provenance(record) == {"clip": {"dog": 0.8}}


def test_legacy_detection_without_model_defaults_to_yolo():
    record = {"detections": [{"label": "cat", "confidence": 0.6}]}
    assert provenance(record) == {"yolo": {"cat": 0.6}}


def test_model_tags_preferred_over_legacy_fields():
    record = {
        "model_tags": {"yolo": {"bird": 0.9}},
        "labels": ["dog"],
        "detections": [{"label": "cat", "confidence": 0.5}],
    }
    assert provenance(record) == {"yolo": {"bird": 0.9}}


# --------------------------------------------------------------------------- rule_confidence


def test_rule_confidence_returns_best_among_fired_signals():
    record = {"model_tags": {"yolo": {"tennis racket": 0.6}, "clip": {"tennis court": 0.85}}}
    assert rule_confidence(record, TENNIS) == 0.85


def test_rule_confidence_zero_when_nothing_matched():
    record = {"model_tags": {"yolo": {"dog": 0.9}}}
    assert rule_confidence(record, TENNIS) == 0.0
