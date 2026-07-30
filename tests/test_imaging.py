from __future__ import annotations

import cv2
import numpy as np

from lib.imaging import draw_boxes, is_ir_frame


def _jpeg(bgr) -> bytes:
    return cv2.imencode(".jpg", bgr)[1].tobytes()


def test_draw_boxes_annotates_and_stays_decodable() -> None:
    base = np.zeros((200, 300, 3), np.uint8)
    out = draw_boxes(_jpeg(base), [("Percy", (10, 20, 120, 160))])
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (200, 300, 3)
    # Something was drawn — the all-black frame now has colored pixels.
    assert decoded.max() > 0


def test_draw_boxes_no_boxes_returns_input_unchanged() -> None:
    data = _jpeg(np.zeros((32, 32, 3), np.uint8))
    assert draw_boxes(data, []) == data


def test_draw_boxes_survives_undecodable_input() -> None:
    assert draw_boxes(b"not a jpeg", [("Percy", (0, 0, 5, 5))]) == b"not a jpeg"


def test_is_ir_frame_detects_grayscale() -> None:
    gray = np.full((64, 64, 3), 120, np.uint8)  # equal channels -> no saturation
    assert is_ir_frame(_jpeg(gray)) is True


def test_is_ir_frame_false_for_colour() -> None:
    color = np.zeros((64, 64, 3), np.uint8)
    color[:, :, 2] = 200  # strong red -> high saturation
    assert is_ir_frame(_jpeg(color)) is False


def test_is_ir_frame_false_on_undecodable() -> None:
    assert is_ir_frame(b"not an image") is False


def test_downscale_array_to_jpeg_caps_long_edge() -> None:
    from lib.imaging import downscale_array_to_jpeg
    frame = np.zeros((1296, 2304, 3), np.uint8)
    out = downscale_array_to_jpeg(frame, max_dim=1024)
    h, w = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert max(h, w) == 1024 and (h, w) == (576, 1024)


def test_is_ir_frame_large_frame_strided_paths_agree() -> None:
    from lib.imaging import is_ir_array

    # Camera-native sizes take the subsampled path; verdicts must match the
    # small-frame path for both IR-ish and colour frames.
    gray = np.full((1296, 2304, 3), 90, np.uint8)
    assert is_ir_array(gray) is True
    color = np.zeros((1296, 2304, 3), np.uint8)
    color[:, :, 2] = 200
    assert is_ir_array(color) is False
