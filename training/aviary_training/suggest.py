"""Shared helpers for the model-assisted annotation suggestion scripts.

The annotation tool's filesystem is its database. Each image already has two
sidecars: ``<image>.json`` (the human source of truth) and ``<image>.txt`` (the
YOLO export). The suggestion scripts add a THIRD, non-authoritative sidecar,
``<image>.suggest.json``, so model output never touches the canonical files:

    {
      "boxModel":  "yolo11n.pt",   # provenance — set by suggest_boxes
      "labelModel": "live-002.pt", # provenance — set by suggest_labels
      "boxes":  [ {id, cx, cy, w, h, conf, label, labelConf} ],  # NEW box proposals
      "labels": [ {boxId, label, conf} ]   # label proposals for EXISTING .json boxes
    }

``suggest_boxes`` owns ``boxes`` (geometry + ``conf``); ``suggest_labels`` owns
``labels`` and fills ``label``/``labelConf`` on any ``boxes`` it matches. Each
script preserves the other's data, so they can run in either order.

This module holds only pure logic (IoU, matching, sidecar IO) so it is shared by
both scripts AND the unit tests without importing ultralytics/torch.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

# Mirror the image set understood by training/scripts/prepare_dataset.py.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Geometry box: a mapping with normalized center+size keys (cx, cy, w, h), the
# same shape as the annotation tool's Box. Helpers below read those four keys.
Box = dict


# --- Geometry ---------------------------------------------------------------


def to_xyxy(box: Box) -> tuple[float, float, float, float]:
    """Normalized center+size -> corner coords (x1, y1, x2, y2)."""
    cx, cy, w, h = float(box["cx"]), float(box["cy"]), float(box["w"]), float(box["h"])
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two normalized center+size boxes (0..1)."""
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_box(detection: Box, candidates: list[Box], threshold: float) -> int | None:
    """Index of the candidate with the highest IoU to ``detection`` above
    ``threshold``, or None when nothing overlaps enough. Ties resolve to the
    earliest (lowest-index) candidate."""
    best_idx: int | None = None
    best_iou = threshold
    for i, cand in enumerate(candidates):
        score = iou(detection, cand)
        if score > best_iou:
            best_iou = score
            best_idx = i
    return best_idx


# --- Sidecar paths ----------------------------------------------------------


def annotation_path(image_path: Path) -> Path:
    """The human source-of-truth sidecar ``<image>.json``."""
    return image_path.with_suffix(".json")


def suggestion_path(image_path: Path) -> Path:
    """The suggestion sidecar ``<image>.suggest.json`` (note the doubled suffix —
    ``with_suffix`` would clobber ``.json``, so build it from the stem)."""
    return image_path.with_name(image_path.stem + ".suggest.json")


# --- JSON IO ----------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomic write (temp + replace), so a reader never sees a half-written file."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def empty_suggestions() -> dict:
    return {"boxes": [], "labels": []}


def read_suggestions(image_path: Path) -> dict:
    """Read ``<image>.suggest.json``, normalized to ``{boxes, labels, ...}``.
    Missing/garbled file -> a fresh empty structure (never raises)."""
    data = _read_json(suggestion_path(image_path))
    if not isinstance(data, dict):
        return empty_suggestions()
    data.setdefault("boxes", [])
    data.setdefault("labels", [])
    if not isinstance(data["boxes"], list):
        data["boxes"] = []
    if not isinstance(data["labels"], list):
        data["labels"] = []
    return data


def write_suggestions(image_path: Path, data: dict) -> None:
    _write_json_atomic(suggestion_path(image_path), data)


def read_annotation_boxes(image_path: Path) -> list[Box]:
    """Boxes from the human ``<image>.json`` (id + geometry + label). Empty list
    when the file is missing or unreadable — suggest_labels still labels the
    sidecar's own box suggestions in that case."""
    data = _read_json(annotation_path(image_path))
    if not isinstance(data, dict):
        return []
    boxes = data.get("boxes")
    if not isinstance(boxes, list):
        return []
    out: list[Box] = []
    for i, b in enumerate(boxes):
        if not isinstance(b, dict):
            continue
        out.append(
            {
                "id": b.get("id", f"b-{i}"),
                "cx": float(b.get("cx", 0)),
                "cy": float(b.get("cy", 0)),
                "w": float(b.get("w", 0)),
                "h": float(b.get("h", 0)),
                "label": b.get("label"),
            }
        )
    return out


def is_boxed(image_path: Path) -> bool:
    """True when a human has confirmed this image's boxes (``boxed: true`` in the
    .json). Used by suggest_boxes' --skip-boxed to avoid re-proposing on done work."""
    data = _read_json(annotation_path(image_path))
    return bool(isinstance(data, dict) and data.get("boxed"))


# --- Discovery --------------------------------------------------------------


def iter_images(source: Path) -> Iterable[Path]:
    """Yield image files under ``source`` (recursive), skipping macOS cruft.
    Mirrors prepare_dataset.iter_images so both walk the exact same file set."""
    for path in sorted(source.rglob("*")):
        if "__MACOSX" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


_MODEL_RE = re.compile(r"-(\d{3})$")


def latest_model(models_dir: Path, prefix: str) -> Path | None:
    """Highest-numbered ``<prefix>-NNN.pt`` under ``models_dir`` (the model the
    train scripts most recently exported), or None if none exist. Mirrors the
    sequence scan in scripts/train_live.{ps1,sh}."""
    best: tuple[int, Path] | None = None
    if not models_dir.is_dir():
        return None
    for path in models_dir.glob(f"{prefix}-*.pt"):
        m = _MODEL_RE.search(path.stem)
        if not m:
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, path)
    return best[1] if best else None
