"""Small image helpers shared across the server.

Camera frames are large (e.g. 2304x1296, ~600 KB JPEG). Uploading them straight
to Telegram times out on a typical home uplink — which silently drops alert,
find-proof and digest photos. Downscaling to a sane size before upload makes
sends fast and reliable while staying perfectly clear on a phone.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


def _label_color(label: str) -> tuple[int, int, int]:
    """A bright, deterministic BGR color per label, so each bird reads the same."""
    hue = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_boxes(
    image_bytes: bytes,
    boxes: list[tuple],
    *,
    quality: int = 85,
) -> bytes:
    """Draw labeled detection rectangles onto a JPEG and re-encode it.

    Each box is ``(label, (x1, y1, x2, y2))`` or ``(label, bbox, color)`` where
    ``color`` is an explicit BGR tuple (e.g. the bird's roster color). Without an
    explicit color a consistent per-label color is derived. Returns the input
    unchanged when there are no boxes or it can't be decoded, so callers can use it
    unconditionally.
    """
    if not boxes:
        return image_bytes
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return image_bytes
    draw_boxes_on_array(array, boxes)
    ok, buffer = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else image_bytes


def draw_boxes_on_array(array, boxes: list[tuple]) -> None:
    """Draw labeled detection rectangles in place on a decoded BGR frame."""
    height, width = array.shape[:2]
    thickness = max(2, round(min(height, width) / 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(height, width) / 900)
    for box in boxes:
        label, (x1, y1, x2, y2) = box[0], box[1]
        color = box[2] if len(box) > 2 and box[2] else _label_color(label or "?")
        cv2.rectangle(array, (x1, y1), (x2, y2), color, thickness)
        if label:
            (text_w, text_h), base = cv2.getTextSize(label, font, font_scale, thickness)
            top = max(text_h + 4, y1)  # keep the label on-screen for a box at the edge
            cv2.rectangle(array, (x1, top - text_h - 4), (x1 + text_w + 4, top + base - 2), color, -1)
            cv2.putText(
                array, label, (x1 + 2, top - 2), font, font_scale,
                (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA,
            )


def draw_boxes_downscaled(
    image_bytes: bytes,
    boxes: list[tuple],
    *,
    max_dim: int = 1280,
    quality: int = 80,
) -> bytes:
    """Resize to ``max_dim`` and draw boxes in ONE decode/encode cycle.

    For VLM-bound frames with no other consumer: drawing at the target size
    (with the box coordinates scaled to match) replaces the draw-then-downscale
    chain — two fewer full JPEG cycles per frame, and the label text is
    rendered at final resolution instead of being resampled. Falls back to the
    input unchanged when the image can't be decoded.
    """
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return image_bytes
    height, width = array.shape[:2]
    scale = max_dim / max(height, width)
    if scale < 1.0:
        array = cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))))
        boxes = [
            (
                box[0],
                tuple(int(round(value * scale)) for value in box[1]),
                *box[2:],
            )
            for box in boxes
        ]
    draw_boxes_on_array(array, boxes)
    ok, buffer = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else image_bytes


def downscale_array_to_jpeg(frame, max_dim: int = 1280, quality: int = 80) -> bytes:
    """JPEG-encode a decoded BGR frame, resizing so the longest edge <= max_dim.

    For callers that already hold the decoded array (the camera/VLM path) — saves
    a redundant JPEG decode versus :func:`downscale_jpeg`.
    """
    height, width = frame.shape[:2]
    scale = max_dim / max(height, width)
    if scale < 1.0:
        # max(1, …) so an extreme aspect ratio can't round the short edge to 0,
        # which would make cv2.resize raise on the photo-send path.
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else b""


def downscale_jpeg(image_bytes: bytes, max_dim: int = 1280, quality: int = 80) -> bytes:
    """Re-encode so the longest edge is <= ``max_dim`` at ``quality``.

    Returns the input unchanged if it can't be decoded or is already small
    enough (a tiny image isn't re-encoded just to shrink the file slightly).
    """
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return image_bytes
    height, width = array.shape[:2]
    if max_dim / max(height, width) >= 1.0 and len(image_bytes) < 200_000:
        # Already small and within size — leave it as-is.
        return image_bytes
    return downscale_array_to_jpeg(array, max_dim, quality) or image_bytes


# Mean HSV saturation below this reads as a grayscale (IR / night-mode) frame.
# Daylight colour frames sit well above it; IR frames are near-zero with a little
# JPEG noise.
IR_SATURATION_THRESHOLD = 16.0


def is_ir_array(frame, threshold: float = IR_SATURATION_THRESHOLD) -> bool:
    """True if a decoded BGR frame looks like night/IR mode (near-grayscale).

    Tapo cameras drop to monochrome IR after dark; the birds can't be told apart
    then, so the server gates colour-dependent work (auto-search) on this.
    Operates on the array the camera loop already has — no JPEG decode — so IR is
    computed once per frame and cached in :class:`lib.ir.IRState`.
    """
    if frame is None:
        return False
    # A mean over ~100k pixels answers "is this grayscale" just as well as one
    # over the full 2304x1296 frame, at a fraction of the HSV-convert cost on
    # the per-inference hot path. Stride keeps small frames untouched.
    stride = max(1, min(frame.shape[0], frame.shape[1]) // 320)
    if stride > 1:
        frame = np.ascontiguousarray(frame[::stride, ::stride])
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(saturation.mean()) < threshold


def is_ir_frame(image_bytes: bytes, threshold: float = IR_SATURATION_THRESHOLD) -> bool:
    """As :func:`is_ir_array` but from JPEG bytes (decodes first). False on a bad
    read — don't assume IR on a frame we couldn't decode."""
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return is_ir_array(array, threshold)
