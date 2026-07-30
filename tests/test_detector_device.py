"""Device auto-pick for multi-GPU boxes (see detector.pick_freest_device)."""

from __future__ import annotations

from lib.detector import pick_freest_device


def test_single_gpu_defers_to_ultralytics() -> None:
    assert pick_freest_device([4_000_000_000]) is None


def test_no_gpu_defers_to_ultralytics() -> None:
    assert pick_freest_device([]) is None


def test_picks_gpu_with_most_free_vram() -> None:
    assert pick_freest_device([100, 4_000_000_000]) == "cuda:1"
    assert pick_freest_device([4_000_000_000, 100]) == "cuda:0"


def test_tie_is_deterministic() -> None:
    assert pick_freest_device([5, 5]) == "cuda:0"
