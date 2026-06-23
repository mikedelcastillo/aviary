from __future__ import annotations

import numpy as np
from PIL import Image

from aviary_training.cut_paste_ir import (
    DAY_TO_SPECIES,
    feather_mask,
    ir_tone,
    overlaps_any,
    parse_boxes,
    paste_object,
    yolo_line,
)


def test_parse_boxes_reads_geometry() -> None:
    assert parse_boxes(["8 0.5 0.5 0.1 0.2", "", "bad", "6 0.1 0.1 0.05 0.05"]) == [
        (0.5, 0.5, 0.1, 0.2),
        (0.1, 0.1, 0.05, 0.05),
    ]


def test_overlaps_any_detects_and_clears() -> None:
    existing = [(0.5, 0.5, 0.2, 0.2)]
    assert overlaps_any((0.5, 0.5, 0.2, 0.2), existing, thr=0.05) is True   # same box
    assert overlaps_any((0.9, 0.9, 0.1, 0.1), existing, thr=0.05) is False  # far away
    assert overlaps_any((0.5, 0.5, 0.2, 0.2), [], thr=0.05) is False        # nothing to hit


def test_day_to_species_mapping() -> None:
    assert DAY_TO_SPECIES == {0: 6, 1: 6, 2: 7, 3: 7, 4: 7, 5: 8}


def test_feather_mask_center_one_corner_low() -> None:
    m = feather_mask(40, 30)
    assert m[15, 20] == 1.0          # center fully opaque
    assert m[0, 0] < 0.3             # corner faded out
    assert m.shape == (30, 40)


def test_ir_tone_is_2d_grayscale_values() -> None:
    img = Image.fromarray(np.full((10, 12, 3), [200, 100, 0], dtype=np.uint8), "RGB")
    g = ir_tone(img)
    assert g.shape == (10, 12)       # single channel
    assert abs(g[0, 0] - (0.5 * 200 + 0.35 * 100 + 0.15 * 0)) < 1e-3


def test_paste_object_box_in_bounds_and_grayscale() -> None:
    rng = np.random.default_rng(0)
    bg = Image.fromarray(rng.integers(0, 256, (400, 600, 3), dtype=np.uint8), "RGB")
    crop = Image.fromarray(rng.integers(0, 256, (60, 50, 3), dtype=np.uint8), "RGB")
    comp, (cx, cy, w, h) = paste_object(bg, crop, 0.5, 0.5, 0.1)
    assert 0 < cx < 1 and 0 < cy < 1 and 0 < w < 1 and 0 < h < 1
    assert cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6  # box within frame
    # the pasted region became grayscale (R==G==B) at the center
    a = np.asarray(comp)
    px = a[200, 300]
    assert px[0] == px[1] == px[2]


def test_paste_object_clamps_near_edge() -> None:
    bg = Image.new("RGB", (300, 300), (40, 40, 40))
    crop = Image.new("RGB", (40, 40), (200, 200, 200))
    _, (cx, cy, w, h) = paste_object(bg, crop, 0.99, 0.99, 0.2)  # pushed off-edge
    assert cx + w / 2 <= 1.0 + 1e-6 and cy + h / 2 <= 1.0 + 1e-6


def test_yolo_line_format() -> None:
    assert yolo_line(8, (0.5, 0.25, 0.1, 0.2)) == "8 0.500000 0.250000 0.100000 0.200000"
