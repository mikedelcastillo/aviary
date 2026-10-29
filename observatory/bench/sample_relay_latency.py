"""Bounded JPEG series from existing local relay sources, preserving input PTS.

Writes private images/metadata only. Never opens raw camera URLs or uses NVENC.
The embedded camera overlay must be read independently; file age is not content age.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from urllib.parse import urlsplit


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def sample(config, camera, output, seconds):
    camera_id = camera["id"]
    folder = output / camera_id
    folder.mkdir()
    command = ["ffmpeg", "-y", "-nostdin", "-v", "error", "-fflags", "nobuffer", "-flags", "low_delay",
        "-rtsp_transport", "tcp", "-timeout", "5000000", "-probesize", "1048576", "-analyzeduration", "500000",
        "-threads", "2", "-i", config["relay_rtsp"] + "/" + camera_id, "-map", "0:v:0", "-an",
        "-vf", "fps=1,scale=1280:720", "-c:v", "mjpeg", "-threads", "1", "-q:v", "2", str(folder / "%04d.jpg")]
    started = time.time()
    first = None
    with (folder / "stderr.private.log").open("wb") as errors:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=errors, start_new_session=True)
        try:
            startup_end = time.monotonic() + 18
            while proc.poll() is None:
                if (folder / "0001.jpg").exists():
                    first = time.time()
                    break
                if time.monotonic() >= startup_end:
                    break
                time.sleep(.05)
            if first is not None:
                until = time.monotonic() + seconds
                while proc.poll() is None and time.monotonic() < until:
                    time.sleep(.1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2)
    finished = time.time()
    images = [{"file": file.name, "write_utc": utc(file.stat().st_mtime), "bytes": file.stat().st_size}
              for file in sorted(folder.glob("*.jpg"))]
    capture = Path(config["runtime"]) / "capture" / (camera_id + ".jpg")
    shutil.copyfile(capture, folder / "capture-at-end.jpg")
    result = {"camera": camera_id, "started_utc": utc(started), "first_output_seen_utc": utc(first) if first else None,
        "finished_utc": utc(finished), "requested_sustained_seconds": seconds, "exit": proc.returncode,
        "images": images, "capture_at_end_copy_utc": utc(time.time()), "capture_file_mtime_utc": utc(capture.stat().st_mtime),
        "method": "Existing loopback relay only; input RTP timestamps preserved; no wallclock rewrite; 1fps output, 2 decoder threads; no NVENC."}
    (folder / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if urlsplit(config["relay_rtsp"]).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("This observer requires an existing loopback relay URL")
    if not 1 <= args.seconds <= 60:
        parser.error("Sustained sample must be 1-60 seconds")
    output = Path(args.output)
    output.mkdir(mode=0o700)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda camera: sample(config, camera, output, args.seconds), config["cameras"]))
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in row.items() if key != "images"} | {"image_count": len(row["images"])} for row in results], indent=2))
