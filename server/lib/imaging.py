"""Small image helpers shared across the server.

Camera frames are large (e.g. 2304x1296, ~600 KB JPEG). Uploading them straight
to Telegram times out on a typical home uplink — which silently drops alert,
find-proof and digest photos. Downscaling to a sane size before upload makes
sends fast and reliable while staying perfectly clear on a phone.
"""

from __future__ import annotations

import cv2
import numpy as np


def downscale_jpeg(image_bytes: bytes, max_dim: int = 1280, quality: int = 80) -> bytes:
    """Re-encode so the longest edge is <= ``max_dim`` at ``quality``.

    Returns the input unchanged if it can't be decoded or is already small
    enough (a tiny image isn't re-encoded just to shrink the file slightly).
    """
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return image_bytes
    height, width = array.shape[:2]
    scale = max_dim / max(height, width)
    if scale < 1.0:
        array = cv2.resize(array, (int(width * scale), int(height * scale)))
    elif len(image_bytes) < 200_000:
        # Already small and within size — leave it as-is.
        return image_bytes
    ok, buffer = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else image_bytes


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
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(saturation.mean()) < threshold


def is_ir_frame(image_bytes: bytes, threshold: float = IR_SATURATION_THRESHOLD) -> bool:
    """As :func:`is_ir_array` but from JPEG bytes (decodes first). False on a bad
    read — don't assume IR on a frame we couldn't decode."""
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return is_ir_array(array, threshold)
