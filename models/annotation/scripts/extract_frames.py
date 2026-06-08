#!/usr/bin/env python3
"""Extract frames from an RTSP stream or video file for annotation."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="RTSP URL or video file")
    parser.add_argument("--output", required=True, type=Path, help="Directory for extracted frames")
    parser.add_argument("--prefix", default="camera", help="Filename prefix")
    parser.add_argument("--every-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def frame_timestamp(capture, cv2, fallback_started_at: float) -> float:
    video_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    if video_msec and video_msec > 0:
        return video_msec / 1000.0
    return time.monotonic() - fallback_started_at


def main() -> None:
    args = parse_args()
    import cv2

    if args.every_seconds <= 0:
        raise SystemExit("--every-seconds must be greater than 0")
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be greater than 0")

    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    saved = 0
    next_save_at = 0.0
    started_at = time.monotonic()
    date_part = datetime.now().strftime("%Y-%m-%d")

    try:
        while saved < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            current_ts = frame_timestamp(capture, cv2, started_at)
            if current_ts < next_save_at:
                continue

            filename = f"{args.prefix}_{date_part}_{saved:05d}.jpg"
            path = args.output / filename
            ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                raise RuntimeError(f"Failed to write {path}")

            saved += 1
            next_save_at = current_ts + args.every_seconds
            print(path)
    finally:
        capture.release()

    print(f"Saved {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
