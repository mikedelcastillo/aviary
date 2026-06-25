from __future__ import annotations

import cv2
import numpy as np

from lib.imaging import is_ir_frame


def _jpeg(bgr) -> bytes:
    return cv2.imencode(".jpg", bgr)[1].tobytes()


def test_is_ir_frame_detects_grayscale() -> None:
    gray = np.full((64, 64, 3), 120, np.uint8)  # equal channels -> no saturation
    assert is_ir_frame(_jpeg(gray)) is True


def test_is_ir_frame_false_for_colour() -> None:
    color = np.zeros((64, 64, 3), np.uint8)
    color[:, :, 2] = 200  # strong red -> high saturation
    assert is_ir_frame(_jpeg(color)) is False


def test_is_ir_frame_false_on_undecodable() -> None:
    assert is_ir_frame(b"not an image") is False
