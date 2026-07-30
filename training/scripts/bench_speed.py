"""Detector speed harness (``uv run bench-speed``).

Measures wall-clock predict latency for the live detector configuration on
this machine's real frames, so any pipeline/device/imgsz change can be proven
faster (or not) with a before/after pair of runs.

  * Defaults mirror the deployed server: model path, confidence, iou, imgsz
    and device come from the same env-resolved ``ModelConfig`` the server
    boots with (``.env`` respected), so a bare ``uv run bench-speed`` scores
    exactly what production runs.
  * Frames are sampled deterministically (seeded) from the collect tree —
    real camera captures at native resolution, not synthetic tensors.
  * Reports warmed-up p50 / p95 / mean latency and imgs/sec, for batch=1 and
    optionally larger batches (``--batch 8``), and appends a JSON record with
    the git revision + device name to ``data/perf/speed.json`` so runs can be
    compared across commits.

Accuracy is deliberately out of scope — that's ``uv run benchmark``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

# cwd-relative like every other script here: `uv run bench-speed` from the repo
# root (the module itself installs into the venv, so __file__ is useless).
DEFAULT_FRAME_DIR = Path("data/server/collect")
DEFAULT_OUT = Path("data/perf/speed.json")


def summarize_latencies(seconds: list[float]) -> dict:
    """p50/p95/mean/imgs-per-sec summary of per-image latencies, in ms."""
    if not seconds:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None, "imgs_per_sec": None}
    ordered = sorted(seconds)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    mean = statistics.fmean(ordered)
    return {
        "count": len(ordered),
        "p50_ms": round(statistics.median(ordered) * 1000, 2),
        "p95_ms": round(ordered[p95_index] * 1000, 2),
        "mean_ms": round(mean * 1000, 2),
        "imgs_per_sec": round(1.0 / mean, 2) if mean > 0 else None,
    }


def sample_frames(frame_dir: Path, count: int, seed: int = 7) -> list[Path]:
    """Deterministic sample of ``count`` jpgs under ``frame_dir``."""
    frames = sorted(frame_dir.rglob("*.jpg"))
    if not frames:
        return []
    rng = random.Random(seed)
    if len(frames) <= count:
        return frames
    return rng.sample(frames, count)


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def append_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            records = []
    records.append(record)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=1))
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector inference speed harness")
    parser.add_argument("--model", help="model path (default: live MODEL_PATH)")
    parser.add_argument("--device", help="cpu / cuda:N / auto (default: live MODEL_DEVICE)")
    parser.add_argument("--imgsz", type=int, help="inference size (default: live MODEL_IMAGE_SIZE)")
    parser.add_argument("--conf", type=float, help="confidence (default: live MODEL_CONFIDENCE)")
    parser.add_argument("--frames", default=str(DEFAULT_FRAME_DIR), help="frame directory")
    parser.add_argument("--count", type=int, default=40, help="timed frames per run")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warmup predicts")
    parser.add_argument("--batch", type=int, default=1, help="images per predict call")
    parser.add_argument("--seed", type=int, default=7, help="frame sampling seed")
    parser.add_argument("--label", default="", help="free-text tag stored with the record")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON results file")
    args = parser.parse_args()

    # Late imports keep --help snappy and let tests import the pure helpers
    # without pulling in torch/cv2/ultralytics.
    import cv2
    from dotenv import load_dotenv

    load_dotenv()

    from lib.config import build_config
    from lib.detector import _resolve_auto_device

    config = build_config().model
    model_path = args.model or config.paths[0]
    device = args.device or config.device
    imgsz = args.imgsz or config.image_size
    conf = args.conf if args.conf is not None else config.confidence

    if device == "auto":
        device = _resolve_auto_device() or "auto(default)"

    frame_paths = sample_frames(Path(args.frames), args.count, args.seed)
    if not frame_paths:
        raise SystemExit(f"no .jpg frames found under {args.frames}")
    frames = [cv2.imread(str(path)) for path in frame_paths]
    frames = [f for f in frames if f is not None]

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predict_args = {"conf": conf, "iou": config.iou, "imgsz": imgsz, "verbose": False}
    if device and "auto" not in device:
        predict_args["device"] = device

    for frame in frames[: args.warmup]:
        model.predict(source=frame, **predict_args)

    latencies: list[float] = []
    detections = 0
    if args.batch <= 1:
        for frame in frames:
            start = time.perf_counter()
            results = model.predict(source=frame, **predict_args)
            latencies.append(time.perf_counter() - start)
            detections += sum(len(r.boxes) for r in results)
    else:
        for index in range(0, len(frames) - args.batch + 1, args.batch):
            chunk = frames[index : index + args.batch]
            start = time.perf_counter()
            results = model.predict(source=chunk, **predict_args)
            elapsed = time.perf_counter() - start
            latencies.extend([elapsed / len(chunk)] * len(chunk))
            detections += sum(len(r.boxes) for r in results)

    summary = summarize_latencies(latencies)
    device_name = device
    try:
        import torch

        if device and device.startswith("cuda"):
            device_name = f"{device} ({torch.cuda.get_device_name(int(device.split(':')[1]))})"
    except Exception:
        pass

    record = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
        "label": args.label,
        "model": str(model_path),
        "device": device_name,
        "imgsz": imgsz,
        "conf": conf,
        "batch": args.batch,
        "warmup": args.warmup,
        "detections": detections,
        **summary,
    }
    append_record(Path(args.out), record)

    print(f"model {model_path}  device {device_name}  imgsz {imgsz}  conf {conf}  batch {args.batch}")
    print(
        f"frames {summary['count']}  p50 {summary['p50_ms']}ms  p95 {summary['p95_ms']}ms  "
        f"mean {summary['mean_ms']}ms  {summary['imgs_per_sec']} imgs/s  detections {detections}"
    )
    print(f"recorded -> {args.out}")


if __name__ == "__main__":
    main()
