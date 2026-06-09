"""Shared test doubles and builders for the immich test suite.

Importable as ``from fakes import ...`` because ``models/immich/tests`` is on pytest's
``pythonpath`` (see the root pyproject's ``[tool.pytest.ini_options]``). Everything here is
dependency-free — no torch/cv2/ultralytics — so the whole suite runs on a plain CPU box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from aviary_immich.models.base import ModelOutput, Tag


class OutOfMemoryError(Exception):
    """Stand-in for ``torch.cuda.OutOfMemoryError``.

    ``records._is_cuda_oom`` matches on the class *name* (``"OutOfMemoryError"``), so naming the
    class this way makes the OOM-halving path in ``inference.predict_batch_adaptive`` fire without
    importing torch.
    """


# --------------------------------------------------------------------------- builders


def make_asset(id: str = "a1", name: str | None = None, path: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build an Immich asset dict. Omit ``name``/``path`` to exercise the id fallback."""
    asset: dict[str, Any] = {"id": id}
    if name is not None:
        asset["originalFileName"] = name
    if path is not None:
        asset["originalPath"] = path
    asset.update(extra)
    return asset


def make_detection(label: str, confidence: float = 0.9, box: tuple[int, int, int, int] = (0, 0, 10, 10)) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {"label": label, "confidence": confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def make_prediction(
    labels: Iterable[str] = (),
    confidence: float = 0.9,
    detections: list[dict[str, Any]] | None = None,
):
    """Build a ``BirdPrediction``. Give ``labels`` for a quick one-detection-per-label result."""
    from aviary_immich.detector import BirdPrediction

    if detections is None:
        detections = [make_detection(label, confidence) for label in labels]
    max_confidence = max((float(detection["confidence"]) for detection in detections), default=0.0)
    return BirdPrediction(has_bird=bool(detections), max_confidence=max_confidence, detections=detections)


def make_record(
    asset_id: str = "a1",
    decision: str = "match",
    labels: list[str] | None = None,
    detections: list[dict[str, Any]] | None = None,
    max_confidence: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    """Build a scan record. Pass ``labels=None`` to exercise the legacy ``detections`` fallback."""
    record: dict[str, Any] = {
        "account": "acct",
        "asset_id": asset_id,
        "decision": decision,
        "max_confidence": max_confidence,
        "original_file_name": f"{asset_id}.jpg",
    }
    if labels is not None:
        record["labels"] = labels
    if detections is not None:
        record["detections"] = detections
    record.update(extra)
    return record


def make_album(name: str = "Birds", id: str = "alb-1", owner_id: str | None = None, **extra: Any) -> dict[str, Any]:
    album: dict[str, Any] = {"id": id, "albumName": name}
    if owner_id is not None:
        album["ownerId"] = owner_id
    album.update(extra)
    return album


# --------------------------------------------------------------------------- fakes


class FakeImmichClient:
    """In-memory stand-in for ``aviary_immich.client.ImmichClient``.

    Records side effects on ``self.added`` (album writes), ``self.created`` (albums created), and
    ``self.downloaded`` (asset ids fetched). ``download_thumbnail``/``download_original`` actually
    write ``thumbnail_bytes`` to the given path so cache/rename logic can be exercised on tmp_path.
    """

    def __init__(
        self,
        *,
        albums: list[dict[str, Any]] | None = None,
        user: dict[str, Any] | None = None,
        album_assets: list[dict[str, Any]] | None = None,
        image_assets: list[dict[str, Any]] | None = None,
        video_assets: list[dict[str, Any]] | None = None,
        thumbnail_bytes: bytes = b"\x89PNGfake",
        thumbnail_content_type: str = "image/jpeg",
        video_bytes: bytes = b"\x00\x00\x00\x18ftypmp4fake",
    ) -> None:
        self._albums = list(albums or [])
        self._user = user if user is not None else {"id": "owner-1", "email": "me@example.com"}
        self._album_assets = list(album_assets or [])
        self._image_assets = list(image_assets or [])
        self._video_assets = list(video_assets or [])
        self._thumbnail_bytes = thumbnail_bytes
        self._thumbnail_content_type = thumbnail_content_type
        self._video_bytes = video_bytes
        self.added: list[tuple[str, list[str]]] = []
        self.created: list[str] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.downloaded: list[str] = []
        self.downloaded_videos: list[str] = []

    def get_my_user(self) -> dict[str, Any]:
        return self._user

    def list_albums(self) -> list[dict[str, Any]]:
        return list(self._albums)

    def create_album(self, album_name: str) -> dict[str, Any]:
        album = {"id": f"created-{album_name}", "albumName": album_name}
        self._albums.append(album)
        self.created.append(album_name)
        return album

    def ensure_owned_album(self, album_name: str) -> dict[str, Any]:
        for album in self._albums:
            if album.get("albumName") == album_name:
                return album
        return self.create_album(album_name)

    def update_album(self, album_id: str, **fields: Any) -> dict[str, Any]:
        self.updated.append((album_id, dict(fields)))
        return {"id": album_id, **fields}

    def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> Any:
        if not asset_ids:
            return []
        self.added.append((album_id, list(asset_ids)))
        return []

    def iter_album_assets(self, album_id: str, page_size: int = 250, limit: int | None = None) -> Iterable[dict[str, Any]]:
        for index, asset in enumerate(self._album_assets):
            if limit is not None and index >= limit:
                return
            yield asset

    def iter_image_assets(self, page_size: int = 250, limit: int | None = None) -> Iterable[dict[str, Any]]:
        for index, asset in enumerate(self._image_assets):
            if limit is not None and index >= limit:
                return
            yield asset

    def iter_video_assets(self, page_size: int = 250, limit: int | None = None) -> Iterable[dict[str, Any]]:
        for index, asset in enumerate(self._video_assets):
            if limit is not None and index >= limit:
                return
            yield asset

    def download_video(self, asset_id: str, path: Path, transcoded: bool = True) -> dict[str, str]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._video_bytes)
        self.downloaded_videos.append(str(asset_id))
        return {"content-type": "video/mp4"}

    def download_thumbnail(self, asset_id: str, path: Path, size: str = "preview") -> dict[str, str]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._thumbnail_bytes)
        self.downloaded.append(str(asset_id))
        return {"content-type": self._thumbnail_content_type}

    def download_original(self, asset_id: str, path: Path) -> dict[str, str]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._thumbnail_bytes)
        self.downloaded.append(str(asset_id))
        return {"content-type": self._thumbnail_content_type}


class FakeDetector:
    """In-memory stand-in for ``PretrainedBirdDetector`` for inference/worker tests.

    ``responder(items) -> list[BirdPrediction]`` controls the result and may raise (e.g. an
    :class:`OutOfMemoryError`) to drive OOM-halving. Each call is appended to ``self.calls`` as
    ``(mode, n_items, batch_size)`` where mode is ``"paths"`` or ``"arrays"``.
    """

    def __init__(self, responder: Callable[[list[Any]], list[Any]] | None = None, half: bool = False) -> None:
        self.responder = responder
        self.half = half
        self.calls: list[tuple[str, int, int]] = []
        self._infer_cap: int | None = None

    def _run(self, items: Iterable[Any], batch_size: int, mode: str) -> list[Any]:
        items = list(items)
        self.calls.append((mode, len(items), batch_size))
        if self.responder is not None:
            return self.responder(items)
        return [make_prediction() for _ in items]

    def predict_batch(self, items: Iterable[Any], batch_size: int = 64) -> list[Any]:
        return self._run(items, batch_size, "paths")

    def predict_batch_arrays(self, items: Iterable[Any], batch_size: int = 64) -> list[Any]:
        return self._run(items, batch_size, "arrays")

    def predict(self, path: Any):
        return self.predict_batch([path], batch_size=1)[0]


# --------------------------------------------------------------------------- model doubles


def make_tag(name: str, confidence: float = 0.9, box: tuple[int, int, int, int] | None = (0, 0, 10, 10)) -> Tag:
    """Build a ``Tag``. Pass ``box=None`` for a scene tag (no bounding box)."""
    return Tag(name=name, confidence=confidence, box=box)


def make_output(
    labels: Iterable[str] = (),
    confidence: float = 0.9,
    tags: Iterable[Tag] | None = None,
    max_confidence: float | None = None,
) -> ModelOutput:
    """Build a ``ModelOutput``. Give ``labels`` for a quick one-tag-per-label result."""
    resolved = tuple(tags) if tags is not None else tuple(make_tag(label, confidence) for label in labels)
    mc = max_confidence if max_confidence is not None else max((tag.confidence for tag in resolved), default=0.0)
    return ModelOutput(tags=resolved, max_confidence=mc)


class FakeModel:
    """In-memory stand-in implementing the ``Model`` protocol for inference/worker/video tests.

    ``responder(items) -> list[ModelOutput]`` controls the result and may raise (e.g. an
    :class:`OutOfMemoryError`) to drive per-model OOM-halving. Each call is appended to
    ``self.calls`` as ``(mode, n_items, batch_size)`` where mode is ``"paths"`` or ``"arrays"``.
    ``name`` is the provenance key written into a record's ``model_tags``.
    """

    def __init__(
        self,
        name: str = "yolo",
        responder: Callable[[list[Any]], list[Any]] | None = None,
        half: bool = False,
    ) -> None:
        self.name = name
        self.responder = responder
        self.half = half
        self.calls: list[tuple[str, int, int]] = []
        self._infer_cap: int | None = None

    def _run(self, items: Iterable[Any], batch_size: int, mode: str) -> list[Any]:
        items = list(items)
        self.calls.append((mode, len(items), batch_size))
        if self.responder is not None:
            return self.responder(items)
        return [make_output() for _ in items]

    def predict_arrays(self, items: Iterable[Any], batch_size: int = 64) -> list[Any]:
        return self._run(items, batch_size, "arrays")

    def predict_paths(self, items: Iterable[Any], batch_size: int = 64) -> list[Any]:
        return self._run(items, batch_size, "paths")
