"""Group-aware, class-stratified train/val/test assignment.

Two problems with a plain random shuffle of CCTV frames:

1. **Leakage.** Near-duplicate frames from the same short motion burst end up in
   different splits, so val/test contain near-twins of train images and the
   reported metrics are optimistic. We prevent this by assigning whole *groups*
   (a camera's frames within a short time window) to a single split.

2. **Unmeasurable rare classes.** A random split leaves scarce classes (e.g. the
   IR species) with near-zero boxes in val and none in test. We spread each label
   across the splits using iterative stratification (Sechidis et al. 2011, the
   algorithm behind scikit-multilearn's IterativeStratification): process labels
   rarest-first and place each group into the split that most needs that label.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

# camera-N_{day,ir}_YYYY-MM-DD_HHMMSS_hash — the Tapo capture naming.
_TAPO_RE = re.compile(r"^(?P<camera>camera-\d+)_(?:day|ir)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{6})_")


def group_key(name: str, group_minutes: int = 10) -> str:
    """A split-grouping key for an image filename.

    Tapo frames group by ``camera + date + time-bucket`` (bucket width
    ``group_minutes``) so a burst of near-duplicate frames shares one key and
    never straddles splits. Phone photos and any other naming have no temporal
    bursts, so each is its own group (keyed by its stem).
    """
    match = _TAPO_RE.match(name)
    if not match:
        return f"solo:{Path(name).stem}"
    hhmmss = match.group("time")
    minute_of_day = int(hhmmss[0:2]) * 60 + int(hhmmss[2:4])
    bucket = minute_of_day // max(1, group_minutes)
    return f"{match.group('camera')}|{match.group('date')}|{bucket}"


def stratified_group_split(
    keys: list[str],
    label_sets: list[list[int]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[str]:
    """Assign each item to ``"train"``/``"val"``/``"test"``.

    ``keys[i]`` is item ``i``'s group key (same key => same split, no leakage);
    ``label_sets[i]`` is its list of class indices. Returns a list of split names
    parallel to ``keys``. Each label is spread across the splits in proportion to
    the ratios as far as the group constraint allows.
    """
    if len(keys) != len(label_sets):
        raise ValueError("keys and label_sets must be the same length")

    ratios = {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}

    members: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        members[key].append(i)

    group_labels: dict[str, Counter] = {}
    label_total: Counter = Counter()
    for key, idxs in members.items():
        counts: Counter = Counter()
        for i in idxs:
            counts.update(label_sets[i])
        group_labels[key] = counts
        label_total.update(counts)

    # "Desired remaining" budgets per split: how many more boxes of each label,
    # and how many more items overall, each split still wants. Greedy assignment
    # drives these toward zero.
    desired_label = {s: {lbl: ratios[s] * tot for lbl, tot in label_total.items()} for s in ratios}
    desired_size = {s: ratios[s] * len(keys) for s in ratios}

    rng = Random(seed)
    groups = list(members.keys())
    rng.shuffle(groups)  # deterministic tie-breaking

    assignment: dict[str, str] = {}

    def place(group: str, split: str) -> None:
        assignment[group] = split
        for lbl, cnt in group_labels[group].items():
            desired_label[split][lbl] -= cnt
        desired_size[split] -= len(members[group])

    # Labeled groups, rarest label first; within a label, the group holding the
    # most of it first (it constrains placement the hardest).
    for lbl in sorted(label_total, key=lambda l: (label_total[l], l)):
        pending = [g for g in groups if g not in assignment and group_labels[g].get(lbl, 0) > 0]
        pending.sort(key=lambda g: -group_labels[g][lbl])
        for group in pending:
            split = max(ratios, key=lambda s: (desired_label[s][lbl], desired_size[s]))
            place(group, split)

    # Label-less groups (confirmed-empty backgrounds): fill remaining capacity.
    for group in groups:
        if group not in assignment:
            place(group, max(ratios, key=lambda s: desired_size[s]))

    return [assignment[key] for key in keys]
