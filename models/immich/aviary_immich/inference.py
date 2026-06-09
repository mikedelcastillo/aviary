"""Adaptive batched detector inference, halving on CUDA OOM so small-VRAM GPUs still run."""

from __future__ import annotations

from typing import Any, Callable

from aviary_immich.records import _is_cuda_oom, chunks, record_from_error, record_from_prediction


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def predict_batch_adaptive(
    detector,
    assets: list[dict[str, Any]],
    items: list[Any],
    account_slug: str,
    use_arrays: bool = False,
) -> list[dict[str, Any]]:
    """Run one batch, halving and retrying on CUDA OOM so small-VRAM GPUs still run.

    ``items`` are either thumbnail paths (``use_arrays=False``) or pre-decoded BGR arrays
    (``use_arrays=True``); both detector entry points share the ``(items, batch_size)`` interface,
    so the OOM halving below is identical for either.

    The first time a size OOMs we remember a smaller cap on the detector (``_infer_cap``) and
    pre-split future batches to it, so we don't repeatedly attempt (and fail) the large size.
    A single image that still OOMs, or any non-OOM failure, is recorded as a per-asset error.
    """
    predict = detector.predict_batch_arrays if use_arrays else detector.predict_batch
    cap = getattr(detector, "_infer_cap", None)
    if cap is not None and len(items) > cap:
        records: list[dict[str, Any]] = []
        for index in range(0, len(items), cap):
            records.extend(
                predict_batch_adaptive(
                    detector, assets[index : index + cap], items[index : index + cap], account_slug, use_arrays
                )
            )
        return records

    oom = False
    try:
        predictions = predict(items, batch_size=len(items))
        if len(predictions) != len(assets):
            raise RuntimeError(f"Detector returned {len(predictions)} results for {len(assets)} images")
        return [record_from_prediction(asset, account_slug, prediction) for asset, prediction in zip(assets, predictions)]
    except Exception as exc:  # noqa: BLE001 - recorded per asset, not fatal
        if _is_cuda_oom(exc) and len(items) > 1:
            oom = True
        else:
            return [record_from_error(asset, account_slug, exc) for asset in assets]

    # The except block has exited, so the exception (and the traceback that pinned the failed
    # forward pass's GPU tensors) is gone — only now can empty_cache actually reclaim it.
    _free_cuda()
    detector._infer_cap = max(1, len(items) // 2)
    mid = len(items) // 2
    return predict_batch_adaptive(detector, assets[:mid], items[:mid], account_slug, use_arrays) + predict_batch_adaptive(
        detector, assets[mid:], items[mid:], account_slug, use_arrays
    )


def scan_asset_batch_with_detector(
    assets_and_items: list[tuple[dict[str, Any], Any]],
    detector,
    account_slug: str,
    inference_batch_size: int,
    progress_advance: Callable[[int], None] | None = None,
    use_arrays: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for batch in chunks(assets_and_items, inference_batch_size):
        assets = [asset for asset, _ in batch]
        items = [item for _, item in batch]
        records.extend(predict_batch_adaptive(detector, assets, items, account_slug, use_arrays))
        if progress_advance is not None:
            progress_advance(len(assets))

    return records
