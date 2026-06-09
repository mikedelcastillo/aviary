"""The model abstraction every detector/classifier implements.

A ``Model`` turns a batch of images into per-image ``ModelOutput`` tags. The image path feeds
pre-decoded BGR uint8 HWC arrays (``predict_arrays``); the multiprocessing CPU worker path feeds
file paths (``predict_paths``). Both mirror the detector's existing array/path entry points so the
adaptive OOM-halving in :mod:`aviary_immich.inference` works unchanged for either.

This module is import-light on purpose (no torch/cv2/ultralytics) so ``rules``, ``records``,
``album_filing`` and the test suite can import the types freely on a plain CPU box. The heavy
backends live in :mod:`aviary_immich.models.yolo` / :mod:`aviary_immich.models.clip` and are
lazy-imported only when a model is actually built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Tag:
    """One signal a model emits for one image.

    ``name`` is the canonical tag (lowercased), e.g. ``"bird"`` or ``"tennis court"`` — the same
    name different models share so AGREEMENT rules can see them concur. ``box`` is the xyxy pixel
    box for object detectors, or ``None`` for whole-image scene classifiers.
    """

    name: str
    confidence: float
    box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ModelOutput:
    """One model's result for one image: the tags it fired plus its highest confidence."""

    tags: tuple[Tag, ...] = ()
    max_confidence: float = 0.0


@runtime_checkable
class Model(Protocol):
    """Structural interface for any model the pipeline runs.

    Implementations are duck-typed (the YOLO/CLIP wrappers and the test ``FakeModel`` all satisfy
    this without inheriting it). ``name`` is the provenance key written into a record's
    ``model_tags`` and referenced by :class:`aviary_immich.rules.Signal`. ``half`` reports whether
    the model runs fp16 (for the device line in the console).
    """

    name: str
    half: bool

    def predict_arrays(self, images: list[Any], batch_size: int) -> list[ModelOutput]:
        """Run on pre-decoded BGR uint8 HWC arrays; one ModelOutput per image, order preserved."""
        ...

    def predict_paths(self, images: list[Any], batch_size: int) -> list[ModelOutput]:
        """Run on image file paths; one ModelOutput per image, order preserved."""
        ...
