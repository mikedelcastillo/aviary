"""Pure domain logic for Immich scan records.

These functions carry no I/O of their own (the timestamp helper aside) and operate purely on
plain dicts, so they are the most directly unit-testable part of the album generator.
"""

from __future__ import annotations

from typing import Any, Iterable


def asset_name(asset: dict[str, Any]) -> str:
    return str(asset.get("originalFileName") or asset.get("originalPath") or asset.get("id") or "")


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def detected_labels(record: dict[str, Any], known: set[str]) -> set[str]:
    """Return the animal categories matched by a record, intersected with ``known``.

    Reads the ``labels`` field written by new scans; falls back to the per-detection labels so
    state files written before multi-category support (which only carry ``detections``) still
    route correctly.
    """
    labels = record.get("labels")
    if labels is None:
        labels = [detection.get("label") for detection in record.get("detections", [])]
    return {str(label).lower() for label in labels if label} & known


def category_confidence(record: dict[str, Any], label: str) -> float:
    """Highest confidence among detections of ``label`` (falls back to the record's overall max)."""
    confidences = [
        float(detection.get("confidence") or 0)
        for detection in record.get("detections", [])
        if str(detection.get("label", "")).lower() == label
    ]
    return max(confidences) if confidences else float(record.get("max_confidence") or 0)


def bump_decision(stats: dict[str, int], record: dict[str, Any]) -> None:
    from aviary_immich.config import ANIMAL_LABELS

    if str(record.get("decision")) == "error":
        stats["errors"] = stats.get("errors", 0) + 1
        return
    labels = detected_labels(record, set(ANIMAL_LABELS))
    if not labels:
        stats["other"] = stats.get("other", 0) + 1
        return
    for label in labels:
        key = f"{label}s"
        stats[key] = stats.get(key, 0) + 1


def scan_postfix(stats: dict[str, int], prefix: str = "") -> str:
    return (
        f"{prefix}birds={stats.get('birds', 0)} dogs={stats.get('dogs', 0)} "
        f"cats={stats.get('cats', 0)} err={stats.get('errors', 0)}"
    )


def record_from_prediction(asset: dict[str, Any], account_slug: str, prediction) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    labels = sorted({str(detection.get("label", "")).lower() for detection in prediction.detections})
    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "match" if labels else "not_match",
        "labels": labels,
        "max_confidence": prediction.max_confidence,
        "detections": prediction.detections,
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def aggregate_video_predictions(
    asset: dict[str, Any], account_slug: str, predictions, frames_scanned: int
) -> dict[str, Any]:
    """Fan many per-frame ``BirdPrediction``s back into one record for a video asset.

    Shaped identically to :func:`record_from_prediction` (so ``handle_scan_record``,
    ``bump_decision``, ``detected_labels`` and ``category_confidence`` all treat videos and
    photos the same) plus ``asset_type``/``frames_scanned`` for observability. ``detections``
    keeps only the single highest-confidence box per label across all frames — enough for album
    filing and ``category_confidence`` without bloating the state JSONL with every frame's boxes.
    """
    from aviary_immich.state import utc_now

    best_by_label: dict[str, dict[str, float | int | str]] = {}
    for prediction in predictions:
        for detection in prediction.detections:
            label = str(detection.get("label", "")).lower()
            confidence = float(detection.get("confidence") or 0)
            current = best_by_label.get(label)
            if current is None or confidence > float(current.get("confidence") or 0):
                best_by_label[label] = detection

    labels = sorted(label for label in best_by_label if label)
    detections = [best_by_label[label] for label in labels]
    max_confidence = max(
        (float(detection.get("confidence") or 0) for detection in detections), default=0.0
    )
    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "match" if labels else "not_match",
        "labels": labels,
        "max_confidence": max_confidence,
        "detections": detections,
        "asset_type": "video",
        "frames_scanned": int(frames_scanned),
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def record_from_error(asset: dict[str, Any], account_slug: str, exc: Exception) -> dict[str, Any]:
    from aviary_immich.state import utc_now

    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "error",
        "error": str(exc),
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
