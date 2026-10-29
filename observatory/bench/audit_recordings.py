"""Read-only inventory and bounded decode of settled development recordings.

Filename/duration coverage is an estimate: constant-rate encoding can conceal
missing source frames through duplication. Keep per-file output private.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def audit(args):
    root = Path(args.runtime) / "recordings"
    started = time.time()
    files = []
    for path in sorted(root.glob("*/*.mp4")):
        try:
            stat = path.stat()
            if started - stat.st_mtime >= 30:
                files.append((path, stat))
        except OSError:
            pass
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {}
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        output.chmod(0o600)
        for path, stat in files:
            camera = path.parent.name
            index = counts.get(camera, 0)
            counts[camera] = index + 1
            row = {"camera": camera, "file": path.name, "size": stat.st_size}
            try:
                row["start"] = datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
                result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                    "format=duration:stream=codec_type,codec_name,avg_frame_rate,sample_rate,channels",
                    "-of", "json", str(path)], capture_output=True, timeout=15)
                row["probe_exit"] = result.returncode
                row["probe_error_lines"] = len(result.stderr.splitlines())
                if result.returncode == 0:
                    probe = json.loads(result.stdout)
                    row["duration"] = float(probe["format"]["duration"])
                    row["streams"] = probe.get("streams", [])
                    if args.decode_every and index % args.decode_every == 0:
                        row["decode_attempted"] = True
                        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                            "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?", "-threads", "1", "-f", "null", "-"],
                            capture_output=True, timeout=60)
                        row["decode_exit"] = result.returncode
                        row["decode_error_lines"] = len(result.stderr.splitlines())
                current = path.stat()
                row["stable"] = (stat.st_size, stat.st_mtime_ns) == (current.st_size, current.st_mtime_ns)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                row["error"] = type(error).__name__
            rows.append(row)
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {"started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time()-started, 3),
        "decode_every": args.decode_every, "cameras": {}}
    for camera in counts:
        items = [row for row in rows if row["camera"] == camera]
        valid = [row for row in items if row.get("probe_exit") == 0 and row.get("stable") and row.get("duration", 0) > 0
                 and any(stream.get("codec_type") == "video" for stream in row.get("streams", []))]
        valid.sort(key=lambda row: row["start"])
        gaps = []
        end = None
        for row in valid:
            if end is not None and row["start"] > end + 2:
                gaps.append({"start_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                    "end_utc": datetime.fromtimestamp(row["start"], timezone.utc).isoformat(),
                    "seconds": round(row["start"]-end, 3)})
            end = max(end or 0, row["start"] + row["duration"])
        summary["cameras"][camera] = {"files_seen": len(items), "valid_probes": len(valid),
            "invalid_or_unstable": len(items)-len(valid),
            "removed_during_scan": sum(row.get("error") == "FileNotFoundError" for row in items),
            "other_check_errors": sum(bool(row.get("error")) and row["error"] != "FileNotFoundError" for row in items),
            "decode_checks": sum("decode_exit" in row for row in items),
            "decode_failures": sum(row.get("decode_exit", 0) != 0 for row in items),
            "first_start_utc": datetime.fromtimestamp(valid[0]["start"], timezone.utc).isoformat() if valid else None,
            "last_end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat() if end else None,
            "gaps_over_2s": gaps,
            "gap_seconds_total": round(sum(gap["seconds"] for gap in gaps), 3)}
    summary["caveat"] = "Retained files only, with 30s settling. Estimated wall-clock coverage from filenames plus media duration; frame duplication and audio quality are not measured. Concurrent retention may remove files."
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-every", type=int, default=30, help="0 disables decoding")
    args = parser.parse_args()
    if args.decode_every < 0:
        parser.error("decode-every must be nonnegative")
    audit(args)
