"""Vision-model helpers: describe or name a camera frame via Ollama.

All calls go through :meth:`OllamaClient.generate` with a base64 image. The
prompts and the name-cleaning are kept here so the finder (scene descriptions),
the camera namer, and the stream narrator share one definition. Network errors
propagate; callers treat vision as best-effort and degrade.
"""

from __future__ import annotations

import base64
import json
import re

from lib.imaging import downscale_jpeg
from lib.labels import pretty


# Downscale frames to this longest edge before sending to the VLM. The empirical
# sweep showed a full 2304px frame takes ~25s while a 1024px one takes ~3s with
# no loss of usefulness (the detection context, below, does the real work).
MAX_VLM_DIM = 1024


# One-sentence "what's happening" used by /find and the stream narrator.
SCENE_PROMPT = (
    "This is a frame from a pet-bird camera. In ONE short sentence under 25 words, say "
    "what the named bird or birds are doing, whether they are together or alone, "
    "and any obvious feeding, play, rest, preening, or health concern. Be concrete. "
    "If no bird is clearly visible, just say so."
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
    pronouns = pronouns or {}
    if not detections:
        return (
            "This is a still from a pet-bird camera. The detector found no birds "
            "in it, but look carefully — a bird may be small or partly hidden."
        )
    parts = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        pronoun = pronouns.get(detection.label.lower())
        # Name + pronoun only — deliberately NO species. The detector already
        # identified the bird, and naming the species makes the model parrot
        # "Percy the lovebird" / "Bambi the budgie", which we don't want.
        who = f"{pretty(detection.label)} ({pronoun})" if pronoun else pretty(detection.label)
        parts.append(f"{who} in the {position_phrase((x1 + x2) / 2, (y1 + y2) / 2, width, height)}")
    return (
        "This is a still from a pet-bird camera. The camera detected these birds: "
        + "; ".join(parts)
        + ". They may be small or far from the camera. The word in parentheses after "
        "a name is that bird's pronoun, for your reference only. Use the detector list "
        "as evidence that those named birds are present; if "
        "a bird is hard to see, say it is hard to see rather than saying no birds are present. "
        "Describe the scene naturally, calling each bird by NAME and using she/he only where it reads "
        "naturally — do NOT write a pronoun in parentheses, and do NOT mention any "
        "species or breed."
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
    max_dim: int | None = MAX_VLM_DIM,
    timeout_seconds: float | None = None,
) -> str:
    # max_dim=None lets a caller that already downscaled the frame skip the
    # re-decode inside describe_image.
    return describe_image(
        client, model, image_bytes, SCENE_PROMPT, context=context, max_dim=max_dim, timeout_seconds=timeout_seconds
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


# -- structured per-bird analysis (v3 memory) -------------------------------

# The canonical single-primary-activity vocabulary the VLM may assign per bird.
# Kept small and concrete so it aggregates cleanly into statistics.
BIRD_ACTIVITIES = (
    "feeding", "drinking", "preening", "resting", "sleeping", "playing",
    "climbing", "exploring", "flying", "bathing", "vocalizing", "alert",
    "interacting", "hidden", "unknown",
)

# Health concerns worth flagging from a photo (coarse — a flag prompts a closer
# look, it is not a diagnosis). Empty/"none" means nothing visibly wrong.
BIRD_HEALTH_FLAGS = ("fluffed", "lethargic", "labored_breathing", "plucking", "injured", "eye_closed")

ANALYZE_PROMPT = (
    "This is a still from a pet-bird camera. Each bird sits inside a colored box "
    "LABELED with its name. Return ONE entry for EVERY labeled box — never skip a bird "
    "and never add a bird that has no box; use the exact name printed on the box. For "
    "each, choose the single most likely CURRENT activity from what you can see (a bird "
    "simply sitting still is 'resting' or 'alert', not 'hidden' — only use 'hidden' if "
    "its box is truly empty/occluded). Give a short posture phrase (e.g. 'perched', 'on "
    "the cage floor', 'hanging on the bars') and any visible health concern, or null if "
    "the bird looks fine. Also give a vivid 1-2 sentence 'scene' — a caretaker's note "
    "of what the birds are doing (feeding, playing, preening, bathing, resting, "
    "interacting or alone), where they are, and anything notable or endearing; be "
    "concrete and specific. CRITICAL: the colored boxes and name labels are just an "
    "overlay to help you — describe the birds directly as if watching them live, and "
    "NEVER mention a box, its color, an outline, or that anything is labeled (write "
    "'Percy preens on a branch', never 'the bird in the green box'). Never use a "
    "species or breed word. Judge only from what is visible."
)


def _analysis_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "scene": {"type": "string"},
            "birds": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": labels},
                        "activity": {"type": "string", "enum": list(BIRD_ACTIVITIES)},
                        "posture": {"type": "string"},
                        # Plain string (a health flag from BIRD_HEALTH_FLAGS, or "none").
                        # A null/enum-mixed type here made some Olla backends emit empty
                        # JSON; analyze_frame normalises "none"/"" to "".
                        "health": {"type": "string"},
                    },
                    "required": ["label", "activity", "health"],
                },
            },
        },
        "required": ["scene", "birds"],
    }


def analyze_frame(
    client,
    model: str,
    image_bytes: bytes,
    labels: list[str],
    *,
    max_dim: int | None = MAX_VLM_DIM,
    timeout_seconds: float | None = None,
) -> dict:
    """Structured per-bird analysis of a LABELED-box frame for the v3 memory.

    ``labels`` are the detector's labels present in the frame; the VLM is
    grammar-constrained (Ollama ``format`` schema) to only those names and the
    canonical activity vocabulary, so activity is attributed to the right bird and
    the output is always valid JSON. Returns ``{"scene": str, "birds": [{label,
    activity, posture, health}]}`` — empty birds on no labels or a bad response.
    """
    labels = sorted({str(label).strip().lower() for label in labels if str(label).strip()})
    if not labels:
        return {"scene": "", "birds": []}
    image = downscale_jpeg(image_bytes, max_dim) if max_dim else image_bytes
    raw = client.generate(
        model,
        ANALYZE_PROMPT,
        images=[encode_image(image)],
        fmt=_analysis_schema(labels),
        timeout_seconds=timeout_seconds,
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"scene": raw.strip()[:400], "birds": []}
    if not isinstance(data, dict):
        return {"scene": "", "birds": []}
    allowed = set(labels)
    birds = []
    for bird in data.get("birds") or []:
        if not isinstance(bird, dict):
            continue
        label = str(bird.get("label", "")).strip().lower()
        if label not in allowed:
            continue  # the model must stick to the boxes it was given
        health = bird.get("health")
        birds.append(
            {
                "label": label,
                "activity": str(bird.get("activity", "")).strip().lower(),
                "posture": str(bird.get("posture", "")).strip(),
                "health": str(health).strip().lower() if health and str(health).lower() != "none" else "",
            }
        )
    return {"scene": str(data.get("scene", "")).strip(), "birds": birds}
