"""Vision-model helpers: describe or name a camera frame via Ollama.

All calls go through :meth:`OllamaClient.generate` with a base64 image. The
prompts and the name-cleaning are kept here so the finder (scene descriptions),
the camera namer, and the stream narrator share one definition. Network errors
propagate; callers treat vision as best-effort and degrade.
"""

from __future__ import annotations

import base64
import re

from lib.imaging import downscale_jpeg
from lib.labels import pretty


# Downscale frames to this longest edge before sending to the VLM. The empirical
# sweep showed a full 2304px frame takes ~25s while a 1024px one takes ~3s with
# no loss of usefulness (the detection context, below, does the real work).
MAX_VLM_DIM = 1024


# One/two-sentence "what's happening" used by /find and the stream narrator.
SCENE_PROMPT = (
    "This is a frame from a pet-bird camera. In ONE or two short sentences, say "
    "what the bird or birds are doing, and whether multiple birds are together. "
    "Be concrete and brief. If no bird is clearly visible, just say so."
)

# A short, unique label for a camera's view (replaces the IP-based name).
CAMERA_NAME_PROMPT = (
    "This is a still from a home camera watching pet birds. Give a SHORT, "
    "descriptive 1-2 word name for the PLACE or main object in frame — name the "
    "thing itself (for example: Window Perch, Food Bowl, Big Cage, Play Gym, "
    "Couch, Desk, Bookshelf, Kitchen Counter). Do NOT use words about the camera "
    "or angle (no 'view', 'viewpoint', 'cam', 'camera', 'cctv', 'vantage', "
    "'angle', 'overhead', 'wide'). Reply with ONLY the name in Title Case — no "
    "punctuation, no quotes, no explanation."
)


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def position_phrase(cx: float, cy: float, width: int, height: int) -> str:
    """Describe a point as top/middle/bottom + left/centre/right thirds."""
    xr = cx / width if width else 0.5
    yr = cy / height if height else 0.5
    horiz = "left" if xr < 0.34 else "right" if xr > 0.66 else "centre"
    vert = "top" if yr < 0.34 else "bottom" if yr > 0.66 else "middle"
    if horiz == "centre" and vert == "middle":
        return "centre"
    return f"{vert}-{horiz}"


def build_detection_context(
    detections,
    width: int,
    height: int,
    species_of: dict[str, str] | None = None,
    pronouns: dict[str, str] | None = None,
) -> str:
    """Turn YOLO detections into grounding text so the VLM knows what to look at.

    The birds are usually tiny in a wide camera frame, so left to itself the VLM
    says "there are no birds". Telling it exactly which birds the detector found,
    where (in thirds), and each one's pronoun makes it describe them accurately
    (and, empirically, much faster). ``detections`` are anything with ``.label``
    and ``.bbox_xyxy``.
    """
    species_of = species_of or {}
    pronouns = pronouns or {}
    if not detections:
        return (
            "This is a still from a pet-bird camera. The detector found no birds "
            "in it, but look carefully — a bird may be small or partly hidden."
        )
    parts = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        label = detection.label.lower()
        bits = []
        pronoun = pronouns.get(label)
        if pronoun:
            bits.append(pronoun)
        species = species_of.get(label)
        if species and species != label:
            bits.append(f"a {species}")
        who = f"{pretty(detection.label)} ({', '.join(bits)})" if bits else pretty(detection.label)
        parts.append(f"{who} in the {position_phrase((x1 + x2) / 2, (y1 + y2) / 2, width, height)}")
    return (
        "This is a still from a pet-bird camera. The camera detected these birds: "
        + "; ".join(parts)
        + ". They may be small or far from the camera. Use each bird's name and "
        "the pronoun given (he/she)."
    )


def describe_image(
    client,
    model: str,
    image_bytes: bytes,
    prompt: str,
    *,
    context: str | None = None,
    max_dim: int = MAX_VLM_DIM,
    timeout_seconds: float | None = None,
) -> str:
    """Run a single image + prompt through the vision model; return its text.

    ``context`` (e.g. :func:`build_detection_context`) is prepended to the prompt
    to ground the model; the image is downscaled to ``max_dim`` for speed.
    """
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    image = downscale_jpeg(image_bytes, max_dim) if max_dim else image_bytes
    return client.generate(
        model,
        full_prompt,
        images=[encode_image(image)],
        timeout_seconds=timeout_seconds,
    ).strip()


def describe_scene(
    client,
    model: str,
    image_bytes: bytes,
    *,
    context: str | None = None,
    timeout_seconds: float | None = None,
) -> str:
    return describe_image(
        client, model, image_bytes, SCENE_PROMPT, context=context, timeout_seconds=timeout_seconds
    )


# Generic words that describe the CAMERA/viewpoint rather than the place itself;
# stripped so names stay descriptive ("Big Cage", not "Big Cage View"/"CCTV").
_REDUNDANT_NAME_WORDS = {
    "view", "viewpoint", "views", "cam", "cams", "camera", "cameras", "cctv",
    "vantage", "angle", "angles", "shot", "feed", "footage", "scene", "area",
    "image", "picture", "photo", "closeup", "close", "perspective", "pov",
    "lens", "capture", "monitor", "surveillance", "security", "overview",
    "overhead", "wide", "frame", "looking", "facing",
}


def clean_camera_name(raw: str) -> str:
    """Normalise a model's free-text into a tidy 1-2 word Title Case label.

    Splits camelCase ("BigCageView" -> "Big Cage View"), drops punctuation and
    generic descriptor words ("view", "cam", ...), keeps the first two
    meaningful words and Title-cases them. "" when nothing usable remains.
    """
    if not raw:
        return ""
    first_line = raw.strip().splitlines()[0]
    # Break camelCase / "BigCage" into separate words before cleaning.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", first_line)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", spaced)
    words = [
        word for word in cleaned.split() if word.lower() not in _REDUNDANT_NAME_WORDS
    ][:2]
    return " ".join(word.capitalize() for word in words)


def name_camera_view(
    client, model: str, image_bytes: bytes, *, timeout_seconds: float | None = None
) -> str:
    """Ask the vision model for a 1-2 word name for a camera's view."""
    raw = describe_image(
        client, model, image_bytes, CAMERA_NAME_PROMPT, timeout_seconds=timeout_seconds
    )
    return clean_camera_name(raw)
