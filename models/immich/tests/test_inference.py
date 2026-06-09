"""Tests for aviary_immich.inference: adaptive OOM-halving batched multi-model inference.

No GPU/torch needed — ``_free_cuda`` no-ops when its torch import fails, and the
``OutOfMemoryError`` double from ``fakes`` is matched by name in ``records._is_cuda_oom``.
"""

from __future__ import annotations

import pytest

from aviary_immich.inference import predict_batch_adaptive, scan_asset_batch_with_models
from fakes import FakeModel, OutOfMemoryError, make_asset, make_output


# --------------------------------------------------------------------------- helpers


def _assets(n: int) -> list[dict]:
    return [make_asset(id=f"a{i}") for i in range(n)]


def _bird_responder(items):
    """One ``bird`` tag per item."""
    return [make_output(labels=["bird"]) for _ in items]


# --------------------------------------------------------------------------- happy path


def test_happy_path_one_record_per_asset_in_order():
    assets = _assets(3)
    items = ["t0", "t1", "t2"]
    model = FakeModel(responder=_bird_responder)

    records = predict_batch_adaptive([model], assets, items, "acct")

    assert len(records) == 3
    assert [r["asset_id"] for r in records] == ["a0", "a1", "a2"]
    assert [r["decision"] for r in records] == ["match", "match", "match"]
    assert all(r["labels"] == ["bird"] for r in records)
    assert all(r["model_tags"] == {"yolo": {"bird": 0.9}} for r in records)


def test_happy_path_single_predict_call_no_halving():
    assets = _assets(3)
    model = FakeModel(responder=_bird_responder)

    predict_batch_adaptive([model], assets, ["t0", "t1", "t2"], "acct")

    assert len(model.calls) == 1
    assert model.calls[0] == ("paths", 3, 3)


def test_happy_path_not_match_when_no_tags():
    assets = _assets(2)
    model = FakeModel(responder=lambda items: [make_output(labels=[]) for _ in items])

    records = predict_batch_adaptive([model], assets, ["t0", "t1"], "acct")

    assert [r["decision"] for r in records] == ["not_match", "not_match"]


# --------------------------------------------------------------------------- two models


def test_two_models_merge_both_tags_into_one_record():
    assets = _assets(2)
    yolo = FakeModel(name="yolo", responder=lambda items: [make_output(labels=["bird"]) for _ in items])
    scene = FakeModel(name="scene", responder=lambda items: [make_output(labels=["beach"]) for _ in items])

    records = predict_batch_adaptive([yolo, scene], assets, ["t0", "t1"], "acct")

    assert len(records) == 2
    for r in records:
        assert r["decision"] == "match"
        # labels is the sorted union of both models' tag names.
        assert r["labels"] == ["beach", "bird"]
        # model_tags carries each model's provenance separately.
        assert r["model_tags"] == {"yolo": {"bird": 0.9}, "scene": {"beach": 0.9}}
        # detections each carry the originating model.
        assert sorted(d["model"] for d in r["detections"]) == ["scene", "yolo"]


# --------------------------------------------------------------------------- OOM halving


def _oom_at_four(items):
    if len(items) == 4:
        raise OutOfMemoryError("CUDA out of memory")
    return [make_output(labels=["bird"]) for _ in items]


def test_oom_halving_splits_four_into_two_and_two():
    assets = _assets(4)
    items = ["t0", "t1", "t2", "t3"]
    model = FakeModel(responder=_oom_at_four)

    records = predict_batch_adaptive([model], assets, items, "acct")

    assert len(records) == 4
    assert [r["asset_id"] for r in records] == ["a0", "a1", "a2", "a3"]
    assert all(r["decision"] == "match" for r in records)


def test_oom_halving_sets_infer_cap_to_two():
    model = FakeModel(responder=_oom_at_four)

    predict_batch_adaptive([model], _assets(4), ["t0", "t1", "t2", "t3"], "acct")

    assert model._infer_cap == 2


def test_oom_halving_call_sizes_are_four_then_two_then_two():
    model = FakeModel(responder=_oom_at_four)

    predict_batch_adaptive([model], _assets(4), ["t0", "t1", "t2", "t3"], "acct")

    sizes = [n for _, n, _ in model.calls]
    assert sizes == [4, 2, 2]


# --------------------------------------------------------------------------- single-image OOM


def test_single_image_oom_records_one_error_no_infinite_recursion():
    assets = _assets(1)
    model = FakeModel(responder=lambda items: (_ for _ in ()).throw(OutOfMemoryError("CUDA out of memory")))

    records = predict_batch_adaptive([model], assets, ["t0"], "acct")

    assert len(records) == 1
    assert records[0]["decision"] == "error"
    assert records[0]["asset_id"] == "a0"
    # Exactly one attempt — single image cannot be halved further.
    assert len(model.calls) == 1


# --------------------------------------------------------------------------- non-OOM exception


def test_non_oom_exception_records_errors_without_halving():
    assets = _assets(3)
    model = FakeModel(responder=lambda items: (_ for _ in ()).throw(ValueError("boom")))

    records = predict_batch_adaptive([model], assets, ["t0", "t1", "t2"], "acct")

    assert len(records) == 3
    assert all(r["decision"] == "error" for r in records)
    assert all(r["error"] == "boom" for r in records)
    # No halving for a non-OOM failure.
    assert len(model.calls) == 1
    assert model._infer_cap is None


# --------------------------------------------------------------------------- count mismatch


def test_count_mismatch_is_caught_as_per_asset_errors():
    assets = _assets(2)
    # Returns 1 output for 2 items -> RuntimeError raised inside, caught, recorded per asset.
    model = FakeModel(responder=lambda items: [make_output(labels=["bird"])])

    records = predict_batch_adaptive([model], assets, ["t0", "t1"], "acct")

    assert len(records) == 2
    assert all(r["decision"] == "error" for r in records)
    assert all("1 results for 2 items" in r["error"] for r in records)
    # RuntimeError is not OOM -> no halving.
    assert len(model.calls) == 1


# --------------------------------------------------------------------------- _infer_cap pre-split


def test_infer_cap_presplit_chunks_five_into_two_two_one():
    assets = _assets(5)
    items = [f"t{i}" for i in range(5)]
    model = FakeModel(responder=_bird_responder)
    model._infer_cap = 2

    records = predict_batch_adaptive([model], assets, items, "acct")

    assert len(records) == 5
    assert [r["asset_id"] for r in records] == ["a0", "a1", "a2", "a3", "a4"]
    sizes = [n for _, n, _ in model.calls]
    assert sizes == [2, 2, 1]
    assert all(n <= 2 for n in sizes)


def test_infer_cap_presplit_preserves_order_and_decisions():
    model = FakeModel(responder=_bird_responder)
    model._infer_cap = 2

    records = predict_batch_adaptive([model], _assets(5), [f"t{i}" for i in range(5)], "acct")

    assert all(r["decision"] == "match" for r in records)


# --------------------------------------------------------------------------- use_arrays toggle


def test_use_arrays_false_uses_predict_paths_mode():
    model = FakeModel(responder=_bird_responder)

    predict_batch_adaptive([model], _assets(2), ["t0", "t1"], "acct", use_arrays=False)

    assert [mode for mode, _, _ in model.calls] == ["paths"]


def test_use_arrays_true_uses_predict_arrays_mode():
    model = FakeModel(responder=_bird_responder)

    predict_batch_adaptive([model], _assets(2), ["t0", "t1"], "acct", use_arrays=True)

    assert [mode for mode, _, _ in model.calls] == ["arrays"]


# --------------------------------------------------------------------------- scan_asset_batch_with_models


def test_scan_batch_chunks_and_concatenates_in_order():
    pairs = [(make_asset(id=f"a{i}"), f"t{i}") for i in range(5)]
    model = FakeModel(responder=_bird_responder)

    records = scan_asset_batch_with_models(pairs, [model], "acct", inference_batch_size=2)

    assert len(records) == 5
    assert [r["asset_id"] for r in records] == ["a0", "a1", "a2", "a3", "a4"]
    # chunked 2 + 2 + 1
    sizes = [n for _, n, _ in model.calls]
    assert sizes == [2, 2, 1]


def test_scan_batch_progress_advance_called_once_per_chunk():
    pairs = [(make_asset(id=f"a{i}"), f"t{i}") for i in range(5)]
    model = FakeModel(responder=_bird_responder)
    advances: list[int] = []

    scan_asset_batch_with_models(
        pairs, [model], "acct", inference_batch_size=2, progress_advance=advances.append
    )

    assert advances == [2, 2, 1]
    assert sum(advances) == 5


def test_scan_batch_progress_advance_optional():
    pairs = [(make_asset(id="a0"), "t0")]
    model = FakeModel(responder=_bird_responder)

    records = scan_asset_batch_with_models(pairs, [model], "acct", inference_batch_size=4)

    assert len(records) == 1


def test_scan_batch_empty_input_returns_empty():
    model = FakeModel(responder=_bird_responder)
    advances: list[int] = []

    records = scan_asset_batch_with_models(
        [], [model], "acct", inference_batch_size=2, progress_advance=advances.append
    )

    assert records == []
    assert advances == []
    assert model.calls == []


def test_scan_batch_forwards_use_arrays():
    pairs = [(make_asset(id="a0"), "t0"), (make_asset(id="a1"), "t1")]
    model = FakeModel(responder=_bird_responder)

    scan_asset_batch_with_models(pairs, [model], "acct", inference_batch_size=4, use_arrays=True)

    assert [mode for mode, _, _ in model.calls] == ["arrays"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
