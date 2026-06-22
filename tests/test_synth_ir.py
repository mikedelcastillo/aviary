from __future__ import annotations

import numpy as np
from PIL import Image

from aviary_training.synth_ir import has_class, remap_to_species, to_ir


def test_remap_individuals_to_species_keeps_geometry() -> None:
    lines = [
        "5 0.5 0.5 0.1 0.2",   # bambi  -> budgie (8)
        "0 0.1 0.1 0.05 0.05",  # draft  -> cockatiel (6)
        "3 0.2 0.2 0.1 0.1",    # matcha -> lovebird (7)
        "8 0.9 0.9 0.1 0.1",    # budgie -> budgie (8, unchanged)
    ]
    assert remap_to_species(lines) == [
        "8 0.5 0.5 0.1 0.2",
        "6 0.1 0.1 0.05 0.05",
        "7 0.2 0.2 0.1 0.1",
        "8 0.9 0.9 0.1 0.1",
    ]


def test_remap_drops_malformed_lines() -> None:
    assert remap_to_species(["", "5 0.5 0.5", "5 0.5 0.5 0.1 0.1"]) == ["8 0.5 0.5 0.1 0.1"]


def test_has_class() -> None:
    lines = ["5 0.5 0.5 0.1 0.1", "2 0.1 0.1 0.1 0.1"]
    assert has_class(lines, 5) is True
    assert has_class(lines, 8) is False


def test_to_ir_is_grayscale_and_brightness_matched() -> None:
    rng = np.random.default_rng(0)
    src = Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), "RGB")
    out = np.asarray(to_ir(src, seed=1, target_mean=90.0), dtype=np.int16)
    # R == G == B everywhere (true grayscale, like real IR). Blur is applied to
    # the single-channel image then broadcast, so channels stay identical.
    assert int((out.max(axis=2) - out.min(axis=2)).max()) == 0
    # Brightness lands near the requested IR mean (noise/clip leave some slack).
    assert 78.0 <= float(out.mean()) <= 102.0


def test_to_ir_is_deterministic_for_a_seed() -> None:
    src = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8), "RGB")
    a = np.asarray(to_ir(src, seed=7))
    b = np.asarray(to_ir(src, seed=7))
    assert np.array_equal(a, b)
