"""Build a v3 structured memory observation from a frame — the ONE place that
turns (image + YOLO detections + VLM analysis) into a :class:`MemoryObservation`.

Both the live capture path (:mod:`lib.memory_maker`) and the backfill migration
(:mod:`lib.memory_migrate`) call this, so a frame saved today and a frame
re-processed from six days ago produce byte-identical structure.

The contract is simple and robust to a flaky VLM: **YOLO is authoritative for WHO**
(one record per detection), and the VLM analysis only *decorates* those records
with activity/posture/health. A bird the VLM forgets to mention still gets a record
(with empty activity) — identity is never lost to a caption miss.
"""

from __future__ import annotations

from typing import Callable

from lib.imaging import draw_boxes
from lib.journal import MemoryDetection, MemoryObservation
from lib.labels import pretty
from lib.rgb.birds import color_bgr

# Callable that runs the structured VLM pass: (image_bytes, labels) -> analysis dict
# shaped like lib.ai.vlm.analyze_frame's return ({"scene": str, "birds": [...]}).
AnalyzeFn = Callable[[bytes, list[str]], dict]


def analysis_has_content(analysis: dict | None) -> bool:
    """Whether a VLM analysis actually says anything (a scene or any bird info).

    A contentless ``{"scene": "", "birds": []}`` reply must be treated like NO
    analysis — stamping provenance on it would permanently mark the observation
    "done" while carrying zero decoration, hiding it from every backfill.
    """
    if not analysis:
        return False
    return bool(str(analysis.get("scene", "")).strip()) or bool(analysis.get("birds"))


def annotate(image_bytes: bytes, detections) -> bytes:
    """Draw each detection's labeled box on the frame, in the bird's roster color.

    Used for the Telegram album and as the VLM's input (the labeled boxes ground the
    VLM's identity). The RAW frame is what gets saved to disk — boxes are derived
    from the stored bboxes, so they can always be redrawn later.
    """
    return draw_boxes(
        image_bytes,
        [(pretty(d.label), tuple(d.bbox_xyxy), color_bgr(d.label)) for d in detections],
    )


def build_observation(
    detections,
    *,
    camera: str,
    photo: str,
    analysis: dict | None,
    detector_model: str = "",
    vlm_model: str = "",
) -> MemoryObservation:
    """Merge YOLO ``detections`` with a VLM ``analysis`` into a v3 observation.

    ``detections`` are objects with ``.label`` / ``.confidence`` / ``.bbox_xyxy``
    (the detector's :class:`~lib.detector.Detection`). ``analysis`` is
    ``analyze_frame``'s output, or None when no VLM pass ran (identity-only record).
    """
    by_label: dict[str, dict] = {}
    scene = ""
    if analysis:
        scene = str(analysis.get("scene", "")).strip()
        for bird in analysis.get("birds", []) or []:
            label = str(bird.get("label", "")).strip().lower()
            if label and label not in by_label:
                by_label[label] = bird

    records: list[MemoryDetection] = []
    for det in detections:
        info = by_label.get(str(det.label).lower(), {})
        activity = str(info.get("activity", "")).strip().lower()
        health = str(info.get("health", "")).strip().lower()
        records.append(
            MemoryDetection(
                label=det.label,
                confidence=float(det.confidence),
                bbox=tuple(int(v) for v in det.bbox_xyxy),
                activity=activity,
                tags=[activity] if activity else [],
                posture=str(info.get("posture", "")).strip(),
                health="" if health in ("", "none", "null") else health,
            )
        )
    return MemoryObservation(
        camera=camera,
        birds=sorted({str(d.label).lower() for d in detections}),
        note=scene,
        photo=photo,
        detections=records,
        detector_model=detector_model,
        # Provenance must reflect what actually ran: stamping the VLM on an
        # observation whose analysis FAILED (Ollama down) — or said nothing —
        # marked outage-era entries as "already done", so no backfill would ever
        # repair them. An empty vlm_model is the durable "needs VLM" marker the
        # backfill scans for.
        vlm_model=vlm_model if analysis_has_content(analysis) else "",
    )
