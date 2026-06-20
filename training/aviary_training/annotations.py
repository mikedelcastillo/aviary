"""Read the annotation tool's ``.json`` sidecars — the label source of truth.

The ``/annotation`` web tool is the SOLE writer of an image's canonical
annotation: an ``<image>.json`` holding ``boxed`` (human-confirmed flag) and
``boxes`` (each with a ``label``). It mirrors a YOLO ``<image>.txt`` export of the
*labeled* boxes only.

Auto-collected frames are different: ``import_collect_birds.py`` drops a
placeholder ``<image>.txt`` with the class hardcoded to ``0`` so the bird is
pre-boxed when you open it — but writes NO ``.json``. Because roster index ``0``
is ``draft``, those un-reviewed placeholders look exactly like hand-labeled
drafts to a tool that reads ``.txt`` blindly. Dataset prep therefore gates on the
``.json`` here so ONLY human-reviewed, fully-labeled frames reach training.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_annotation(image: Path) -> dict | None:
    """Parse an image's ``.json`` sidecar, or return ``None`` if absent/corrupt.

    A missing sidecar is the normal case for an un-reviewed auto-collected frame;
    a corrupt one is treated the same (un-reviewed) rather than raising, so a
    single bad file never aborts a dataset build.
    """
    sidecar = image.with_suffix(".json")
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def review_status(image: Path) -> str:
    """Classify an image by how far through human review it is.

    - ``"unreviewed"`` — no (or corrupt) ``.json``: an auto placeholder or an
      image never opened. NOT safe to train on.
    - ``"unboxed"``   — a ``.json`` exists but ``boxed`` is not true: seeded/opened
      but not human-confirmed.
    - ``"incomplete"`` — ``boxed`` but at least one box still has no label (still
      in the label queue). Training it would teach the model that the
      un-exported bird is background.
    - ``"ready"``     — human-confirmed and every box labeled (including the
      legitimate empty "nothing here" negative).
    """
    data = load_annotation(image)
    if data is None:
        return "unreviewed"
    if data.get("boxed") is not True:
        return "unboxed"
    boxes = data.get("boxes") or []
    if not all(str(box.get("label") or "").strip() for box in boxes):
        return "incomplete"
    return "ready"


def is_training_ready(image: Path) -> bool:
    """True iff the image is human-confirmed and fully labeled (see ``review_status``)."""
    return review_status(image) == "ready"
