"""Verified bird-care knowledge for the flock, and retrieval over it.

``facts`` holds the curated, fact-checked data (care facts, toxic foods, the
sleep/temperature numbers); ``knowledge`` selects the facts relevant to a
message and formats them as a grounding block for the chat/Q&A path.
"""

from __future__ import annotations

from lib.care.facts import (
    CARE_FACTS,
    CARE_SUMMARY,
    COLD_DANGER_F,
    COLD_STRESS_F,
    FRESH_FOOD_MINUTES,
    SLEEP_HOURS,
    SLEEP_HOURS_TEXT,
    TEMPERATURE_RANGE_TEXT,
    TOXIC_FOODS,
    CareFact,
    ToxicFood,
)
from lib.care.knowledge import (
    care_context,
    care_overview,
    care_reply,
    care_tip,
    detect_species,
    detected_topics,
    relevant_facts,
    toxic_food_in,
    toxic_food_list,
)

__all__ = [
    "CARE_FACTS",
    "CARE_SUMMARY",
    "COLD_DANGER_F",
    "COLD_STRESS_F",
    "FRESH_FOOD_MINUTES",
    "SLEEP_HOURS",
    "SLEEP_HOURS_TEXT",
    "TEMPERATURE_RANGE_TEXT",
    "TOXIC_FOODS",
    "CareFact",
    "ToxicFood",
    "care_context",
    "care_overview",
    "care_reply",
    "care_tip",
    "detect_species",
    "detected_topics",
    "relevant_facts",
    "toxic_food_in",
    "toxic_food_list",
]
