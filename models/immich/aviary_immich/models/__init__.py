"""The model layer: the shared ``Model`` types plus the factory that builds them from config.

Re-exports the import-light base types so callers can ``from aviary_immich.models import Model,
ModelOutput, Tag`` without reaching into the submodule. :func:`build_models` turns the declarative
:class:`~aviary_immich.config.ModelSpec` list into live model instances, lazy-importing each heavy
backend (YOLO/CLIP) only for the kinds actually requested.
"""

from __future__ import annotations

from typing import Any

from aviary_immich.models.base import Model, ModelOutput, Tag


def build_models(
    specs: Any, device: str, default_model: str, default_threshold: float
) -> list:
    """Instantiate every enabled :class:`~aviary_immich.config.ModelSpec`, order preserved.

    YOLO specs fall back to ``default_threshold`` when their own ``threshold`` is ``None``; CLIP
    specs bring their own. Each backend is imported lazily inside its branch so building a
    YOLO-only pipeline never imports CLIP (and vice versa).
    """
    models: list = []
    for spec in specs:
        if not spec.enabled:
            continue
        if spec.kind == "yolo":
            from aviary_immich.models.yolo import YoloObjectModel

            threshold = spec.threshold if spec.threshold is not None else default_threshold
            models.append(YoloObjectModel(default_model, threshold, device, spec.labels))
        elif spec.kind == "clip":
            from aviary_immich.models.clip import ClipSceneModel

            models.append(
                ClipSceneModel(
                    prompts=spec.prompts,
                    threshold=spec.threshold,
                    device=device,
                    **spec.options,
                )
            )
        else:
            raise ValueError(f"unknown model kind: {spec.kind}")
    return models


__all__ = ["Model", "ModelOutput", "Tag", "build_models"]
