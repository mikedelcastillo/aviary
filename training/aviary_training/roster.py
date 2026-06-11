"""Single-source-of-truth bird roster.

One YAML lists every label across both models; each label is tagged with which
model(s) it feeds (``live`` / ``archive``). The order of labels defines the
integer class index used in the YOLO label exports, so a given model's class map is a
*filtered, remapped* view of the full roster. See ``models/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RosterLabel:
    name: str
    index: int  # global class index = position in the roster = YOLO export index
    models: tuple[str, ...]


@dataclass(frozen=True)
class ModelClasses:
    names: dict[int, str]  # contiguous new index -> name (for dataset.yaml)
    remap: dict[int, int]  # global index -> contiguous new index


def load_roster(path: Path) -> list[RosterLabel]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    roster: list[RosterLabel] = []
    for index, entry in enumerate(data.get("labels", [])):
        roster.append(
            RosterLabel(
                name=str(entry["name"]),
                index=index,
                models=tuple(entry.get("models", [])),
            )
        )
    return roster


def classes_for_model(roster: list[RosterLabel], model: str) -> ModelClasses:
    selected = [label for label in roster if model in label.models]
    names = {new: label.name for new, label in enumerate(selected)}
    remap = {label.index: new for new, label in enumerate(selected)}
    return ModelClasses(names=names, remap=remap)


def remap_label_lines(lines: Iterable[str], remap: dict[int, int]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        cls = int(parts[0])
        if cls not in remap:
            continue
        parts[0] = str(remap[cls])
        out.append(" ".join(parts))
    return out
