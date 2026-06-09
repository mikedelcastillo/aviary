"""Pure domain logic for Immich scan records.

These functions carry no I/O of their own (the timestamp helper aside) and operate purely on
plain dicts, so they are the most directly unit-testable part of the album generator.
"""

from __future__ import annotations

from typing import Any, Iterable

from aviary_immich.models.base import ModelOutput, Tag


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
    """Tally one record into per-album counters (keyed by album name) or ``errors``/``other``.

    Routes through the album rules so the live console totals match the manifest exactly: a record
    that matches multiple albums bumps each one. Imports the rules/config lazily to keep this module
    free of the (heavier) config import at module load.
    """
    from aviary_immich.config import ALBUM_RULES
    from aviary_immich.rules import evaluate_rules

    if str(record.get("decision")) == "error":
        stats["errors"] = stats.get("errors", 0) + 1
        return
    matched = evaluate_rules(record, ALBUM_RULES)
    if not matched:
        stats["other"] = stats.get("other", 0) + 1
        return
    for name in matched:
        stats[name] = stats.get(name, 0) + 1


def scan_postfix(stats: dict[str, int], prefix: str = "") -> str:
    """Render the live progress postfix: one ``album=count`` per album (in config order) plus errors."""
    from aviary_immich.config import album_names

    counts = " ".join(f"{name}={stats.get(name, 0)}" for name in album_names())
    return f"{prefix}{counts} err={stats.get('errors', 0)}"


def model_output_from_prediction(prediction) -> ModelOutput:
    """Adapt a legacy ``BirdPrediction`` (xyxy detections) into a :class:`ModelOutput`.

    Each detection becomes a boxed :class:`Tag` with a lowercased label, preserving the bridge from
    the YOLO detector's dict-shaped output to the model-agnostic tag stream the records now build on.
    """
    tags = tuple(
        Tag(
            str(detection.get("label", "")).lower(),
            float(detection.get("confidence") or 0),
            (
                int(detection.get("x1") or 0),
                int(detection.get("y1") or 0),
                int(detection.get("x2") or 0),
                int(detection.get("y2") or 0),
            ),
        )
        for detection in prediction.detections
    )
    return ModelOutput(tags=tags, max_confidence=float(prediction.max_confidence or 0))


def record_from_outputs(
    asset: dict[str, Any], account_slug: str, outputs: dict[str, ModelOutput]
) -> dict[str, Any]:
    """Build the canonical scan record from one image's per-model :class:`ModelOutput`s.

    ``model_tags`` carries the provenance the rules read (``{model: {tag: max confidence}}``), while
    the flat ``labels``/``detections``/``max_confidence`` fields are retained for back-compat. Scene
    tags without a box still contribute to ``model_tags`` and ``labels`` but are omitted from the
    boxed ``detections`` list.
    """
    from aviary_immich.state import utc_now

    model_tags: dict[str, dict[str, float]] = {}
    detections: list[dict[str, Any]] = []
    label_set: set[str] = set()
    max_confidence = 0.0

    for model_name, output in outputs.items():
        tags = output.tags
        if not tags:
            continue
        per_tag: dict[str, float] = {}
        for tag in tags:
            per_tag[tag.name] = max(per_tag.get(tag.name, 0.0), tag.confidence)
            label_set.add(tag.name)
            if tag.box is not None:
                x1, y1, x2, y2 = tag.box
                detections.append(
                    {
                        "label": tag.name,
                        "confidence": tag.confidence,
                        "model": model_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        model_tags[model_name] = per_tag
        max_confidence = max(max_confidence, output.max_confidence)

    labels = sorted(label_set)
    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "match" if labels else "not_match",
        "labels": labels,
        "max_confidence": max_confidence,
        "detections": detections,
        "model_tags": model_tags,
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def record_from_prediction(asset: dict[str, Any], account_slug: str, prediction) -> dict[str, Any]:
    """Single-model (legacy YOLO) record: adapt the prediction and delegate to ``record_from_outputs``."""
    return record_from_outputs(asset, account_slug, {"yolo": model_output_from_prediction(prediction)})


def aggregate_video_outputs(
    asset: dict[str, Any],
    account_slug: str,
    frame_outputs: list[dict[str, ModelOutput]],
    frames_scanned: int,
) -> dict[str, Any]:
    """Fan many per-frame, per-model :class:`ModelOutput`s back into one record for a video asset.

    Shaped identically to :func:`record_from_outputs` (so the rules, ``bump_decision`` and the
    filing logic treat videos and photos the same) plus ``asset_type``/``frames_scanned`` for
    observability. For each ``(model, tag)`` only the single highest-confidence occurrence across
    all frames is kept — enough for album filing without bloating the state JSONL with every frame.
    """
    from aviary_immich.state import utc_now

    # (model, tag) -> winning Tag across all frames
    best: dict[tuple[str, str], Tag] = {}
    best_max_confidence: dict[str, float] = {}
    for outputs in frame_outputs:
        for model_name, output in outputs.items():
            if output.max_confidence > best_max_confidence.get(model_name, 0.0):
                best_max_confidence[model_name] = output.max_confidence
            for tag in output.tags:
                key = (model_name, tag.name)
                current = best.get(key)
                if current is None or tag.confidence > current.confidence:
                    best[key] = tag

    model_tags: dict[str, dict[str, float]] = {}
    detections: list[dict[str, Any]] = []
    label_set: set[str] = set()
    max_confidence = 0.0

    for (model_name, tag_name), tag in best.items():
        model_tags.setdefault(model_name, {})[tag_name] = tag.confidence
        label_set.add(tag_name)
        if tag.box is not None:
            x1, y1, x2, y2 = tag.box
            detections.append(
                {
                    "label": tag_name,
                    "confidence": tag.confidence,
                    "model": model_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    for model_name in model_tags:
        max_confidence = max(max_confidence, best_max_confidence.get(model_name, 0.0))

    labels = sorted(label_set)
    return {
        "account": account_slug,
        "asset_id": str(asset["id"]),
        "decision": "match" if labels else "not_match",
        "labels": labels,
        "max_confidence": max_confidence,
        "detections": detections,
        "model_tags": model_tags,
        "asset_type": "video",
        "frames_scanned": int(frames_scanned),
        "original_file_name": asset_name(asset),
        "scanned_at": utc_now(),
    }


def aggregate_video_predictions(
    asset: dict[str, Any], account_slug: str, predictions, frames_scanned: int
) -> dict[str, Any]:
    """Single-model (legacy YOLO) video aggregation: adapt each frame and delegate."""
    frame_outputs = [{"yolo": model_output_from_prediction(p)} for p in predictions]
    return aggregate_video_outputs(asset, account_slug, frame_outputs, frames_scanned)


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
