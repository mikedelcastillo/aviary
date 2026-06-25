"""Vision-model helpers: describe or name a camera frame via Ollama.

All calls go through :meth:`OllamaClient.generate` with a base64 image. The
prompts and the name-cleaning are kept here so the finder (scene descriptions),
the camera namer, and the stream narrator share one definition. Network errors
propagate; callers treat vision as best-effort and degrade.
"""

from __future__ import annotations

import base64
import re


# One/two-sentence "what's happening" used by /find and the stream narrator.
SCENE_PROMPT = (
    "This is a frame from a pet-bird camera. In ONE or two short sentences, say "
    "what the bird or birds are doing, and whether multiple birds are together. "
    "Be concrete and brief. If no bird is clearly visible, just say so."
)

# A short, unique label for a camera's view (replaces the IP-based name).
CAMERA_NAME_PROMPT = (
    "This is a still from a home camera watching pet birds. Give a SHORT, unique "
    "1-2 word label for THIS camera's view, based on the most distinctive thing "
    "in frame (for example: Window Perch, Food Bowl, Big Cage, Play Gym, Couch, "
    "Desk). Reply with ONLY the label in Title Case — no punctuation, no quotes, "
    "no explanation."
)


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def describe_image(
    client,
    model: str,
    image_bytes: bytes,
    prompt: str,
    *,
    timeout_seconds: float | None = None,
) -> str:
    """Run a single image + prompt through the vision model; return its text."""
    return client.generate(
        model,
        prompt,
        images=[encode_image(image_bytes)],
        timeout_seconds=timeout_seconds,
    ).strip()


def describe_scene(client, model: str, image_bytes: bytes, *, timeout_seconds: float | None = None) -> str:
    return describe_image(client, model, image_bytes, SCENE_PROMPT, timeout_seconds=timeout_seconds)


def clean_camera_name(raw: str) -> str:
    """Normalise a model's free-text into a tidy 1-2 word Title Case label.

    Takes the first line, drops anything but letters/digits/spaces, keeps the
    first two words, Title-cases them. Returns "" when nothing usable remains.
    """
    if not raw:
        return ""
    first_line = raw.strip().splitlines()[0]
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", first_line)
    words = cleaned.split()[:2]
    return " ".join(word.capitalize() for word in words)


def name_camera_view(
    client, model: str, image_bytes: bytes, *, timeout_seconds: float | None = None
) -> str:
    """Ask the vision model for a 1-2 word name for a camera's view."""
    raw = describe_image(
        client, model, image_bytes, CAMERA_NAME_PROMPT, timeout_seconds=timeout_seconds
    )
    return clean_camera_name(raw)
