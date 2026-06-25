from __future__ import annotations

import cv2
import numpy as np

from lib.imaging import downscale_jpeg


def _jpeg(width: int, height: int) -> bytes:
    array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", array)
    assert ok
    return buffer.tobytes()


def _dims(image_bytes: bytes) -> tuple[int, int]:
    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return array.shape[1], array.shape[0]  # (width, height)


def test_downscales_large_frame_to_max_dim() -> None:
    big = _jpeg(2304, 1296)
    out = downscale_jpeg(big, max_dim=1280)
    w, h = _dims(out)
    assert max(w, h) == 1280
    assert len(out) < len(big)


def test_leaves_small_image_untouched() -> None:
    small = _jpeg(320, 240)
    # Well under both max_dim and the size threshold -> returned as-is.
    assert downscale_jpeg(small, max_dim=1280) == small


def test_invalid_bytes_returned_unchanged() -> None:
    assert downscale_jpeg(b"not an image") == b"not an image"
