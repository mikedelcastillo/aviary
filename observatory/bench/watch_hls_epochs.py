"""Capture natural HLS recorder restart boundaries without changing services."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


def packets(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_packets", "-show_entries", "packet=pts_time", "-of", "json", str(path)],
        capture_output=True, timeout=15)
    points = [float(row["pts_time"]) for row in json.loads(result.stdout or b"{}").get("packets", []) if "pts_time" in row]
    return {"probe_exit": result.returncode, "error_lines": len(result.stderr.splitlines()),
            "first_pts": points[0] if points else None, "last_pts": points[-1] if points else None, "packets": len(points)}


def run(args):
    root = Path(args.runtime)
    output = Path(args.output)
    output.mkdir(mode=0o700)
    previous = {}
    boundaries = []
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and len(boundaries) < args.stop_after:
        for playlist in sorted((root / "hls").glob("*/live.m3u8")):
            camera = playlist.parent.name
            try:
                state = json.loads((root / "recorder" / f"{camera}.json").read_text())
                started = state["started"]
                stat = playlist.stat()
                if stat.st_mtime < started:
                    continue  # Old cached output is not output from the new child.
                text = playlist.read_text()
                segments = [line for line in text.splitlines() if line and not line.startswith("#")]
                if not segments or any(Path(name).name != name for name in segments):
                    continue
                old = previous.get(camera)
                if old and old["started"] != started:
                    folder = output / f"{len(boundaries)+1:02}-{camera}"
                    folder.mkdir()
                    (folder / "before.m3u8").write_text(old["manifest"])
                    (folder / "after.m3u8").write_text(text)
                    shutil.copyfile(playlist.parent / old["last_segment"], folder / "before.ts")
                    shutil.copyfile(playlist.parent / segments[0], folder / "after.ts")
                    before, after = packets(folder / "before.ts"), packets(folder / "after.ts")
                    row = {"camera": camera, "observed_utc": datetime.now(timezone.utc).isoformat(),
                        "before_started": old["started"], "after_started": started,
                        "before_segment": old["last_segment"], "after_segment": segments[0],
                        "before": before, "after": after,
                        "after_discontinuity_tags": text.splitlines().count("#EXT-X-DISCONTINUITY"),
                        "after_discontinuity_sequence": next((line.split(":", 1)[1] for line in text.splitlines() if line.startswith("#EXT-X-DISCONTINUITY-SEQUENCE:")), None)}
                    if before["last_pts"] is not None and after["first_pts"] is not None:
                        row["video_pts_step"] = after["first_pts"] - before["last_pts"]
                    (folder / "result.json").write_text(json.dumps(row, indent=2) + "\n")
                    boundaries.append(row)
                    print(json.dumps(row), flush=True)
                previous[camera] = {"started": started, "last_segment": segments[-1], "manifest": text}
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                continue  # Source startup/rolling-file races are not observer failures.
        time.sleep(1)
    (output / "summary.json").write_text(json.dumps({"boundaries": boundaries,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Natural restart; copied latest pre-restart TS and first post-restart TS; no source/service changes."}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--stop-after", type=int, default=3)
    args = parser.parse_args()
    if args.seconds <= 0 or args.stop_after <= 0:
        parser.error("Duration and boundary count must be positive")
    run(args)
