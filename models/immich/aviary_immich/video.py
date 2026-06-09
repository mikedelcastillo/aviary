"""Frame sampling and the per-asset scan loop for Immich VIDEO assets.

Videos break the image pipeline's ``1 asset == 1 frame == 1 prediction == 1 record`` contract:
each video fans out to N sampled frames and fans back in to a single record. They are also a
minority of assets whose cost is dominated by media download + decode (not GPU forward passes),
so they get this dedicated, simpler path rather than threading fan-in through ``run_gpu_pipeline``.
Frames within a single video are still GPU-batched (and OOM-halved) per model, so fp16 and
small-VRAM safety carry over from the image path.

``cv2`` is imported lazily inside the sampling glue so importing this module — and the CLI that
wires it — never pulls the GPU/CV stack (the ``test_cli`` import guarantee).
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from aviary_immich.album_filing import AlbumTarget, handle_scan_record
from aviary_immich.records import (
    aggregate_video_outputs,
    bump_decision,
    record_from_error,
    scan_postfix,
)
from aviary_immich.workers import thread_local_client_factory


# --------------------------------------------------------------------------- pure helpers


def sample_interval(duration_seconds: float, every_seconds: float, max_frames: int) -> float:
    """Seconds between kept frames so ``max_frames`` spreads across the whole video.

    Uses ``every_seconds`` as a floor (don't sample finer than asked) but stretches the spacing to
    ``duration / max_frames`` when the video is long enough that the frame budget wouldn't otherwise
    reach the end — so a 10-minute clip with a 30-frame budget samples across all 10 minutes rather
    than only its first 30 seconds.
    """
    every = max(0.0, every_seconds)
    if duration_seconds > 0 and max_frames > 0:
        return max(every, duration_seconds / max_frames)
    return every


def target_size(height: int, width: int, max_edge: int | None) -> tuple[int, int] | None:
    """Return ``(width, height)`` to downscale to so the long edge ≤ ``max_edge``, or None.

    None means "leave the frame as-is" (already small enough, or no cap requested). Keeping per-frame
    resolution near the image path's "preview" thumbnail bounds GPU memory and decode cost.
    """
    if not max_edge or max_edge <= 0:
        return None
    long_edge = max(height, width)
    if long_edge <= max_edge:
        return None
    scale = max_edge / long_edge
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def select_frames(read_next: Callable[[], tuple[bool, Any, float]], max_frames: int, interval: float) -> list[Any]:
    """Keep one frame per ``interval`` seconds, up to ``max_frames``.

    ``read_next() -> (ok, frame, timestamp_seconds)`` is the only I/O dependency, so this selection
    loop is unit-testable without cv2. ``ok=False`` ends the stream. Frames whose timestamp has not
    yet reached the next sample point are skipped.
    """
    frames: list[Any] = []
    next_at = 0.0
    while max_frames <= 0 or len(frames) < max_frames:
        ok, frame, timestamp = read_next()
        if not ok:
            break
        if timestamp < next_at:
            continue
        frames.append(frame)
        next_at = timestamp + interval
    return frames


# --------------------------------------------------------------------------- cv2 glue


def sample_video_frames(
    video_path: Path,
    every_seconds: float = 1.0,
    max_frames: int = 30,
    max_edge: int | None = 1280,
) -> list[Any]:
    """Decode ``video_path`` and return up to ``max_frames`` BGR uint8 HWC arrays.

    Mirrors the ``cv2.VideoCapture(..., cv2.CAP_FFMPEG)`` + ``CAP_PROP_POS_MSEC`` pattern in
    ``models/annotation/scripts/extract_frames.py``. The returned arrays are exactly what
    ``detector.predict_batch_arrays`` consumes — no path round-trip. Sequential reads (not frame
    seeks) keep this robust across codecs.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    interval = sample_interval(duration, every_seconds, max_frames)

    index = {"value": 0}

    def read_next() -> tuple[bool, Any, float]:
        ok, frame = capture.read()
        if not ok:
            return False, None, 0.0
        msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        if msec and msec > 0:
            timestamp = msec / 1000.0
        elif fps > 0:
            timestamp = index["value"] / fps
        else:
            timestamp = float(index["value"])
        index["value"] += 1
        return True, frame, timestamp

    try:
        frames = select_frames(read_next, max_frames, interval)
    finally:
        capture.release()

    resized: list[Any] = []
    for frame in frames:
        size = target_size(frame.shape[0], frame.shape[1], max_edge)
        resized.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if size else frame)
    return resized


# --------------------------------------------------------------------------- inference fan-in


def predict_frames_adaptive(models, frames: list[Any], batch_size: int) -> list[dict[str, Any]]:
    """Run each model over one video's frames, then transpose into one dict per frame.

    Each model runs over *all* frames through ``inference._run_model_adaptive`` (the shared helper
    that halves on CUDA OOM via the model's ``_infer_cap``, the same logic the image path uses), so
    fp16 and small-VRAM safety carry over. The per-model ``list[ModelOutput]`` results — each aligned
    to ``frames`` — are transposed so ``frame_outputs[i]`` maps ``model.name -> ModelOutput`` for
    frame ``i``, ready to fan into a single record. A non-OOM failure propagates so the caller
    records one asset error.
    """
    from aviary_immich.inference import _run_model_adaptive

    frames = list(frames)
    if not frames:
        return []

    per_model = {model.name: _run_model_adaptive(model, list(frames), True) for model in models}
    return [{name: per_model[name][i] for name in per_model} for i in range(len(frames))]


# --------------------------------------------------------------------------- scan loop


def _video_suffix(asset: dict[str, Any]) -> str:
    name = str(asset.get("originalFileName") or "")
    suffix = Path(name).suffix.lower()
    return suffix if suffix else ".mp4"


def run_video_pipeline(
    video_assets: list[dict[str, Any]],
    models,
    client,
    base_url: str,
    api_key: str,
    account,
    targets: list[AlbumTarget],
    args: argparse.Namespace,
    state: dict[str, dict[str, Any]],
    state_appender,
    stats: dict[str, int],
) -> None:
    """Download → sample → infer → aggregate → file each video, deleting the media after.

    Downloads and frame decode run on a bounded pool of ``download_workers`` threads (requests and
    cv2 both release the GIL) a few videos ahead of the single-threaded models, so inference is
    not stalled on the network. Each finished record flows through the *same* state write +
    ``bump_decision`` + ``handle_scan_record`` calls the image path uses, so videos file into the
    same albums and the caller's final ``flush_album_batch`` handles both. Temp videos are never
    cached (they are large): each is unlinked as soon as its frames are decoded into memory.
    """
    from aviary_immich.console import make_scan_progress

    client_factory = thread_local_client_factory(base_url, api_key)
    video_dir = args.cache_dir / account.slug / "videos"
    inference_batch_size = max(1, args.inference_batch_size)
    workers = max(1, min(args.download_workers, len(video_assets)))

    progress = make_scan_progress()
    total = len(video_assets)
    task: dict[str, Any] = {"id": None}

    def prepare(asset: dict[str, Any]) -> tuple[dict[str, Any], list[Any] | None, BaseException | None]:
        worker_client = client_factory()
        asset_id = str(asset["id"])
        temp_path = video_dir / f"{asset_id}{_video_suffix(asset)}"
        try:
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            worker_client.download_video(asset_id, temp_path, transcoded=args.video_transcoded)
            frames = sample_video_frames(
                temp_path, args.video_every_seconds, args.video_max_frames, args.video_max_edge
            )
            return asset, frames, None
        except BaseException as exc:  # noqa: BLE001 - recorded per asset, not fatal
            return asset, None, exc
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def handle(asset: dict[str, Any], frames: list[Any] | None, error: BaseException | None) -> None:
        if error is not None:
            record = record_from_error(asset, account.slug, error)
        else:
            try:
                frame_outputs = predict_frames_adaptive(models, frames or [], inference_batch_size)
                record = aggregate_video_outputs(asset, account.slug, frame_outputs, len(frames or []))
            except Exception as exc:  # noqa: BLE001 - one error record per video
                record = record_from_error(asset, account.slug, exc)
        state_appender.write(record)
        state[str(record["asset_id"])] = record
        bump_decision(stats, record)
        stats["videos"] = stats.get("videos", 0) + 1
        handle_scan_record(record, asset, account, client, targets, args.dry_run, args.batch_size, stats)
        if progress is not None:
            progress.update(task["id"], advance=1, postfix=scan_postfix(stats))

    def run() -> None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            asset_iter = iter(video_assets)
            inflight = set()
            for _ in range(workers):
                nxt = next(asset_iter, None)
                if nxt is None:
                    break
                inflight.add(executor.submit(prepare, nxt))
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    handle(*future.result())
                    nxt = next(asset_iter, None)
                    if nxt is not None:
                        inflight.add(executor.submit(prepare, nxt))

    if progress is None:
        run()
    else:
        with progress:
            task["id"] = progress.add_task("video", total=total, postfix=scan_postfix(stats))
            run()
