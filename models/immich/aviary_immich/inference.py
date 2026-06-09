"""Adaptive batched model inference, halving on CUDA OOM so small-VRAM GPUs still run."""

from __future__ import annotations

from typing import Any, Callable

from aviary_immich.records import _is_cuda_oom, chunks, record_from_error, record_from_outputs


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_model_adaptive(model, items: list[Any], use_arrays: bool) -> list[Any]:
    """Run ONE model over ``items``, halving and retrying on CUDA OOM.

    Returns the model's ``list[ModelOutput]`` (one per item). The first time a size OOMs we
    remember a smaller cap on the model (``_infer_cap``) and pre-split future batches to it, so
    we don't repeatedly attempt (and fail) the large size. Any non-OOM exception is re-raised.
    """
    cap = getattr(model, "_infer_cap", None)
    if cap is not None and len(items) > cap:
        outputs: list[Any] = []
        for index in range(0, len(items), cap):
            outputs.extend(_run_model_adaptive(model, items[index : index + cap], use_arrays))
        return outputs

    predict = model.predict_arrays if use_arrays else model.predict_paths
    oom = False
    try:
        outputs = predict(items, batch_size=len(items))
        if len(outputs) != len(items):
            raise RuntimeError(f"Model {model.name} returned {len(outputs)} results for {len(items)} items")
        return outputs
    except Exception as exc:  # noqa: BLE001 - OOM is retried by halving; others re-raised
        if _is_cuda_oom(exc) and len(items) > 1:
            oom = True
        else:
            raise

    # Critically, free + recurse only AFTER the except block has exited: while ``except ... as exc``
    # is live, ``exc`` (and its traceback) pins the failed forward pass's GPU tensors, so
    # ``empty_cache`` there reclaims nothing and the halving cascade leaks until even size 1 OOMs.
    _free_cuda()
    model._infer_cap = max(1, len(items) // 2)
    mid = len(items) // 2
    return _run_model_adaptive(model, items[:mid], use_arrays) + _run_model_adaptive(model, items[mid:], use_arrays)


def predict_batch_adaptive(
    models: list[Any],
    assets: list[dict[str, Any]],
    items: list[Any],
    account_slug: str,
    use_arrays: bool = False,
) -> list[dict[str, Any]]:
    """Run every model over one batch, halving and retrying on CUDA OOM, then merge per asset.

    ``items`` are either thumbnail paths (``use_arrays=False``) or pre-decoded BGR arrays
    (``use_arrays=True``); both model entry points share the ``(items, batch_size)`` interface,
    so the per-model OOM halving is identical for either.

    A non-OOM failure from any model is recorded as a per-asset error for the whole batch
    (same coarse semantics as before).
    """
    try:
        per_model: dict[str, list[Any]] = {}
        for model in models:
            per_model[model.name] = _run_model_adaptive(model, items, use_arrays)
    except Exception as exc:  # noqa: BLE001 - recorded per asset, not fatal
        return [record_from_error(asset, account_slug, exc) for asset in assets]

    return [
        record_from_outputs(asset, account_slug, {name: per_model[name][i] for name in per_model})
        for i, asset in enumerate(assets)
    ]


def scan_asset_batch_with_models(
    assets_and_items: list[tuple[dict[str, Any], Any]],
    models: list[Any],
    account_slug: str,
    inference_batch_size: int,
    progress_advance: Callable[[int], None] | None = None,
    use_arrays: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for batch in chunks(assets_and_items, inference_batch_size):
        assets = [asset for asset, _ in batch]
        items = [item for _, item in batch]
        records.extend(predict_batch_adaptive(models, assets, items, account_slug, use_arrays))
        if progress_advance is not None:
            progress_advance(len(assets))

    return records
