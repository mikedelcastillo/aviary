from __future__ import annotations

from collections import Counter

from aviary_training.splits import group_key, stratified_group_split


# --- group_key --------------------------------------------------------------

def test_tapo_frames_in_same_camera_date_bucket_share_a_group() -> None:
    a = group_key("camera-2_day_2026-06-12_014400_aaaa.jpg", group_minutes=10)
    b = group_key("camera-2_day_2026-06-12_014755_bbbb.jpg", group_minutes=10)  # +3m, same bucket
    assert a == b


def test_tapo_frames_in_different_time_buckets_differ() -> None:
    a = group_key("camera-2_day_2026-06-12_014400_aaaa.jpg", group_minutes=10)
    b = group_key("camera-2_day_2026-06-12_023000_bbbb.jpg", group_minutes=10)  # +46m
    assert a != b


def test_different_cameras_never_share_a_group() -> None:
    a = group_key("camera-2_day_2026-06-12_014400_aaaa.jpg", group_minutes=10)
    b = group_key("camera-3_day_2026-06-12_014400_bbbb.jpg", group_minutes=10)
    assert a != b


def test_ir_frames_group_like_day_frames() -> None:
    a = group_key("camera-2_ir_2026-06-12_014400_aaaa.jpg", group_minutes=10)
    b = group_key("camera-2_ir_2026-06-12_014401_bbbb.jpg", group_minutes=10)
    assert a == b


def test_phone_and_unrecognized_names_are_their_own_group() -> None:
    a = group_key("phone_2022-11-04_uuid-1.jpg")
    b = group_key("phone_2022-11-04_uuid-2.jpg")
    assert a != b  # each phone photo is independent (no temporal bursts)


# --- stratified_group_split -------------------------------------------------

def test_items_in_the_same_group_never_straddle_splits() -> None:
    keys = ["a", "a", "a", "b", "c", "d", "e", "f", "g", "h"]
    label_sets = [[0]] * len(keys)
    split = stratified_group_split(keys, label_sets, val_ratio=0.2, test_ratio=0.2, seed=0)
    a_splits = {s for s, k in zip(split, keys) if k == "a"}
    assert len(a_splits) == 1


def test_zero_val_and_test_puts_everything_in_train() -> None:
    keys = [f"g{i}" for i in range(5)]
    label_sets = [[0]] * 5
    split = stratified_group_split(keys, label_sets, val_ratio=0.0, test_ratio=0.0, seed=0)
    assert set(split) == {"train"}


def test_a_label_spread_across_many_groups_reaches_all_three_splits() -> None:
    keys = [f"g{i}" for i in range(20)]
    label_sets = [[7]] * 20  # the rare IR class, but present in 20 distinct groups
    split = stratified_group_split(keys, label_sets, val_ratio=0.2, test_ratio=0.2, seed=0)
    counts = Counter(split)
    assert counts["train"] >= 1 and counts["val"] >= 1 and counts["test"] >= 1
    assert sum(counts.values()) == 20


def test_is_deterministic_for_a_seed() -> None:
    keys = [f"g{i % 7}" for i in range(20)]
    label_sets = [[i % 3] for i in range(20)]
    a = stratified_group_split(keys, label_sets, val_ratio=0.2, test_ratio=0.1, seed=42)
    b = stratified_group_split(keys, label_sets, val_ratio=0.2, test_ratio=0.1, seed=42)
    assert a == b
