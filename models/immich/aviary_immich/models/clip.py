"""CLIP zero-shot scene classifier.

Object detectors (YOLO/COCO) can see a *tennis racket* but not a *tennis court* — a court is a
scene, not an object. This model fills that gap: it scores each image against a set of natural-
language prompts and emits a canonical :class:`~aviary_immich.models.base.Tag` whenever the cosine
similarity clears a threshold. Tennis unions this with the YOLO racket signal so a photo of a court
(or a match) with no clearly visible racket still lands in the album.

Heavy imports (``open_clip``, ``torch``, ``PIL``, ``numpy``) are lazy — this module imports only the
light :mod:`aviary_immich.models.base`, so it can be imported without the CLIP stack present. The
backend is only pulled when a :class:`ClipSceneModel` is actually constructed, which happens only
when a CLIP :class:`~aviary_immich.config.ModelSpec` is enabled (off by default; ``IMMICH_CLIP=1``).
Install it with ``uv sync --group clip`` (or ``uv pip install open_clip_torch``) — it rides whatever
per-machine torch build is already in the venv.

CLIP cosine similarities are NOT on YOLO's confidence scale (they cluster much lower and vary per
prompt), so ``threshold`` must be calibrated separately — see the phase-2 verification in the plan.
"""

from __future__ import annotations

from typing import Any

from aviary_immich.models.base import ModelOutput, Tag


class ClipSceneModel:
    """Zero-shot scene tagger satisfying the :class:`~aviary_immich.models.base.Model` protocol.

    ``prompts`` maps a canonical tag name to the text prompts that detect it (averaged into one
    text embedding per tag). A tag fires for an image when ``cosine(image, tag) >= threshold``.
    """

    name = "clip"

    def __init__(
        self,
        prompts: dict[str, tuple[str, ...]],
        threshold: float | None,
        device: str,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        import open_clip
        import torch

        from aviary_immich.detector import _should_use_fp16, select_device

        self.name = "clip"
        self.threshold = 0.28 if threshold is None else float(threshold)
        self.device = select_device(device)
        # Reuse the detector's fp16 policy verbatim: Pascal/CPU stay fp32, Ampere+/ROCm go fp16,
        # honoring IMMICH_BIRD_HALF — so CLIP behaves identically across the GPU matrix.
        self.half = _should_use_fp16(self.device)
        self._infer_cap: int | None = None
        self._tag_names = list(prompts)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        if self.half:
            self.model = self.model.half()
        tokenizer = open_clip.get_tokenizer(model_name)
        self._text_features = self._encode_prompts(prompts, tokenizer, torch)

    def _encode_prompts(self, prompts: dict[str, tuple[str, ...]], tokenizer, torch) -> Any:
        """Normalized text embedding per tag (the tag's prompts averaged), as an (n_tags, dim) tensor."""
        features = []
        with torch.no_grad():
            for tag in self._tag_names:
                tokens = tokenizer(list(prompts[tag])).to(self.device)
                embedding = self.model.encode_text(tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                features.append(embedding.mean(dim=0))
        stacked = torch.stack(features)
        return stacked / stacked.norm(dim=-1, keepdim=True)

    def predict_arrays(self, images: list[Any], batch_size: int) -> list[ModelOutput]:
        return self._predict(images, batch_size, decoded=True)

    def predict_paths(self, images: list[Any], batch_size: int) -> list[ModelOutput]:
        return self._predict(images, batch_size, decoded=False)

    def _predict(self, images: list[Any], batch_size: int, decoded: bool) -> list[ModelOutput]:
        import numpy as np
        import torch
        from PIL import Image

        images = list(images)
        if not images:
            return []

        outputs: list[ModelOutput] = []
        step = max(1, batch_size)
        for start in range(0, len(images), step):
            chunk = images[start : start + step]
            tensors = []
            for image in chunk:
                if decoded:
                    # Decoded frames/thumbnails are BGR uint8 HWC (cv2); CLIP wants RGB. The
                    # ascontiguousarray copy fixes the negative stride from the ::-1 flip.
                    pil = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
                else:
                    pil = Image.open(image).convert("RGB")
                tensors.append(self.preprocess(pil))
            batch = torch.stack(tensors).to(self.device)
            if self.half:
                batch = batch.half()
            with torch.no_grad():
                features = self.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                similarities = features @ self._text_features.T
            for row in similarities.float().cpu().tolist():
                tags = tuple(
                    Tag(self._tag_names[index], float(score), None)
                    for index, score in enumerate(row)
                    if score >= self.threshold
                )
                outputs.append(
                    ModelOutput(tags=tags, max_confidence=max((tag.confidence for tag in tags), default=0.0))
                )
        return outputs
