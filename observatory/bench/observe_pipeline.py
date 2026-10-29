"""Read-only development-host observation; never restarts application services.

Keep output private: it contains deployment camera IDs and recording filenames.
Configuration credentials, API payloads, and free-form FFmpeg errors are omitted.
The report is sampled evidence, not proof of uninterrupted client playback.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request


DEFAULT_UNITS = "aviary-home,aviary-adapter,aviary-relay,aviary-vision,aviary-recorder,aviary-sync,aviary-curator,aviary-catalog"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def age(path, now):
    return round(now - path.stat().st_mtime, 3) if path.exists() else None


def read_json(path):
    return json.loads(path.read_text())


def proc_rss(pid):
    try:
        status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
        return int(status.get("VmRSS", "0").split()[0])
    except (OSError, ValueError):
        return 0


def units_snapshot(units):
    result = subprocess.run(["systemctl", "--user", "show", *units,
        "-p", "Id", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID",
        "-p", "MemoryCurrent", "-p", "MemoryPeak", "-p", "MemoryMax", "-p", "MemoryHigh",
        "-p", "NRestarts", "-p", "ControlGroup"],
        capture_output=True, text=True, timeout=10, check=True)
    output = {}
    for block in result.stdout.strip().split("\n\n"):
        values = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        unit = values.pop("Id", "unknown")
        group = values.pop("ControlGroup", "")
        cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
        values["main_rss_kib"] = proc_rss(values.get("MainPID", "0"))
        try:
            if not group:
                raise OSError("Unit has no control group")
            pids = set()
            for file in cgroup.rglob("cgroup.procs"):
                pids.update(file.read_text().split())
            values["cgroup_rss_sum_kib"] = sum(proc_rss(pid) for pid in pids)
            memory = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
            values["memory_anon_bytes"] = int(memory.get("anon", 0))
            values["memory_file_bytes"] = int(memory.get("file", 0))
            values["memory_events"] = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
            values["memory_swap_bytes"] = int((cgroup / "memory.swap.current").read_text())
            values["pressure"] = {
                kind: {line.split()[0]: dict(item.split("=", 1) for item in line.split()[1:])
                       for line in (cgroup / (kind + ".pressure")).read_text().splitlines()}
                for kind in ["memory", "cpu", "io"]
            }
        except OSError:
            values["cgroup_read_error"] = True
        output[unit] = values
    return output


def playlist_snapshot(path, now):
    result = {"age_seconds": age(path, now)}
    if not path.exists():
        return result
    lines = path.read_text().splitlines()
    entries = [line for line in lines if line and not line.startswith("#")]
    result["segment_count"] = len(entries)
    result["missing_segments"] = sum(not (path.parent / line).is_file() for line in entries)
    result["sequence"] = next((int(line.split(":", 1)[1]) for line in lines if line.startswith("#EXT-X-MEDIA-SEQUENCE:")), None)
    dates = [line.split(":", 1)[1] for line in lines if line.startswith("#EXT-X-PROGRAM-DATE-TIME:")]
    if dates:
        result["latest_program_age_seconds"] = round(now - datetime.fromisoformat(dates[-1].replace("Z", "+00:00")).timestamp(), 3)
    result["discontinuities"] = lines.count("#EXT-X-DISCONTINUITY")
    return result


def probe_clip(path, decode_seconds):
    before = path.stat()
    result = {"file": path.name, "size": before.st_size, "full_clip_decode": decode_seconds == 0,
              "mtime_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
              "mtime_age_seconds": round(time.time()-before.st_mtime, 3)}
    started = time.monotonic()
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,avg_frame_rate,sample_rate,channels,duration,nb_frames",
        "-of", "json", str(path)], capture_output=True, timeout=20)
    result["probe_exit"] = probe.returncode
    result["probe_error_lines"] = len(probe.stderr.splitlines())
    if probe.returncode == 0:
        result["probe"] = json.loads(probe.stdout)
        # One decoder thread bounds the observer's CPU footprint. Full output is
        # discarded; decoding checks corruption, not perceptual audio quality.
        command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
            "-map", "0:v:0", "-map", "0:a:0?", "-threads", "1"]
        if decode_seconds > 0:
            command += ["-t", str(decode_seconds)]
        decoded = subprocess.run(command + ["-f", "null", "-"], capture_output=True, timeout=45)
        result["decode_exit"] = decoded.returncode
        result["decode_error_lines"] = len(decoded.stderr.splitlines())
    after = path.stat()
    result["stable_during_check"] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    result["check_seconds"] = round(time.monotonic() - started, 3)
    return result


def camera_snapshot(root, camera_id, now, do_probe, decode_seconds):
    # Previous cameras may spend seconds decoding. Timestamp each camera's own
    # observation to avoid reporting negative ages from a stale shared clock.
    now = time.time()
    result = {}
    for source, folder, keys in [
        ("vision", "frames", ["reconnects", "inference_ms", "privacy", "zone_motion_pause"]),
        ("recorder", "recorder", ["started", "playlist_age", "restarts", "encoder"]),
    ]:
        try:
            data = read_json(root / folder / f"{camera_id}.json")
            result[source] = {key: data[key] for key in keys if key in data}
            result[source]["heartbeat_age_seconds"] = round(now - data["timestamp"], 3)
            if source == "recorder":
                result[source]["bounded_error_count"] = len(data.get("errors", []))
                for key in ["child_pid", "output_started", "output_idle_seconds"]:
                    if key in data:
                        result[source][key] = data[key]
                reasons = {None, "shutdown", "process-exit", "memory-emergency", "memory-limit", "startup-timeout", "output-stalled"}
                result[source]["stop_reason"] = data.get("stop_reason") if data.get("stop_reason") in reasons else "other"
                memory = data.get("memory") or {}
                result[source]["child_memory"] = {
                    key: memory[key] for key in ["rss_bytes", "anonymous_bytes", "swap_bytes", "private_bytes"]
                    if isinstance(memory.get(key), (int, float))
                }
                progress = data.get("progress") or {}
                safe_progress = {}
                for key in ["frame", "out_time_us"]:
                    try:
                        safe_progress[key] = int(progress[key])
                    except (KeyError, TypeError, ValueError):
                        pass
                try:
                    safe_progress["speed"] = float(str(progress["speed"]).removesuffix("x"))
                except (KeyError, TypeError, ValueError):
                    pass
                if progress.get("progress") in {"continue", "end"}:
                    safe_progress["state"] = progress["progress"]
                result[source]["progress"] = safe_progress
        except (OSError, ValueError, KeyError) as error:
            result[source] = {"read_error": type(error).__name__}
    try:
        result["playlist"] = playlist_snapshot(root / "hls" / camera_id / "live.m3u8", now)
    except (OSError, ValueError) as error:
        result["playlist"] = {"read_error": type(error).__name__}
    clips = []
    for path in (root / "recordings" / camera_id).glob("*.mp4"):
        try:
            clips.append((path, path.stat()))
        except OSError:
            pass
    clips.sort(key=lambda entry: entry[0].name)
    settled = [(path, stat) for path, stat in clips if now - stat.st_mtime >= 30]
    result["recordings"] = {"count": len(clips), "bytes": sum(stat.st_size for _, stat in clips),
        "newest_mtime_age_seconds": round(now - max(stat.st_mtime for _, stat in clips), 3) if clips else None,
        "newest_settled_file": settled[-1][0].name if settled else None}
    if do_probe and settled:
        try:
            result["clip_check"] = probe_clip(settled[-1][0], decode_seconds)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result["clip_check"] = {"error": type(error).__name__}
    return result


def catalog_snapshot(root, now):
    now = time.time()
    result = {}
    try:
        data = read_json(root / "catalog.json")
        result = {key: data.get(key) for key in ["indexed", "invalid", "probed", "scan_seconds"]}
        result["heartbeat_age_seconds"] = round(now - data["timestamp"], 3)
    except (OSError, ValueError, KeyError) as error:
        result["heartbeat_error"] = type(error).__name__
    with sqlite3.connect(f"file:{root / 'aviary.sqlite'}?mode=ro", uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "clips" in tables:
            result["clips_by_status"] = dict(db.execute("SELECT status,count(*) FROM clips GROUP BY status"))
            result["latest_clip_time"] = db.execute("SELECT max(time) FROM clips").fetchone()[0]
        if "recording_failures" in tables:
            result["known_invalid_recordings"] = db.execute("SELECT count(*) FROM recording_failures").fetchone()[0]
        if "events" in tables:
            result["event_count"] = db.execute("SELECT count(*) FROM events").fetchone()[0]
    return result


def run(args):
    config = read_json(Path(args.config))
    root = Path(config["runtime"])
    cameras = [item["id"] for item in config["cameras"]]
    units = args.units.split(",")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental appending to an unrelated run.
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        output.chmod(0o600)
        started = time.monotonic()
        deadline = started + args.seconds
        due = started
        next_probe = started
        index = 0
        while True:
            now = time.time()
            row = {"schema": 1, "timestamp": utc_now(), "index": index,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "planned_seconds": args.seconds, "interval_seconds": args.interval,
                "decode_seconds": args.decode_seconds}
            try:
                request_started = time.monotonic()
                with urllib.request.urlopen(args.health_url, timeout=5) as response:
                    row["api"] = {"status": response.status, "milliseconds": round((time.monotonic() - request_started) * 1000, 3)}
            except (OSError, urllib.error.URLError) as error:
                row["api"] = {"error": type(error).__name__}
            do_probe = time.monotonic() >= next_probe
            row["cameras"] = {camera: camera_snapshot(root, camera, now, do_probe, args.decode_seconds) for camera in cameras}
            if do_probe:
                next_probe = time.monotonic() + args.probe_interval
            for label, function in [("units", lambda: units_snapshot(units)), ("catalog", lambda: catalog_snapshot(root, now))]:
                try:
                    row[label] = function()
                except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
                    row[label] = {"error": type(error).__name__}
            row["collection_seconds"] = round(time.monotonic() - started - row["elapsed_seconds"], 3)
            row["complete"] = time.monotonic() >= deadline
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if row["complete"]:
                break
            index += 1
            due += args.interval
            time.sleep(max(0, min(due, deadline) - time.monotonic()))
    print(json.dumps({"samples": index + 1, "completed": utc_now(), "planned_seconds": args.seconds}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=10800)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--probe-interval", type=float, default=300)
    parser.add_argument("--decode-seconds", type=float, default=0, help="0 decodes the complete settled clip")
    parser.add_argument("--health-url", default="http://127.0.0.1:8740/api/health")
    parser.add_argument("--units", default=DEFAULT_UNITS)
    args = parser.parse_args()
    if args.seconds < 0 or args.interval <= 0 or args.probe_interval <= 0 or args.decode_seconds < 0:
        parser.error("durations must be nonnegative; intervals must be positive")
    if not all(math.isfinite(value) for value in [args.seconds, args.interval, args.probe_interval, args.decode_seconds]):
        parser.error("durations must be finite")
    run(args)
