"""Summarize private JSONL from observe_pipeline.py without claiming continuity."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return {"count": 0}
    return {"count": len(values), "median": round(statistics.median(values), 3),
            "p95": values[int((len(values)-1)*.95)], "max": values[-1]}


def summarize(rows):
    if not rows:
        raise ValueError("No samples")
    result = {"samples": len(rows), "start": rows[0]["timestamp"], "end": rows[-1]["timestamp"],
        "elapsed_seconds": rows[-1]["elapsed_seconds"], "planned_seconds": rows[-1]["planned_seconds"],
        "complete": rows[-1].get("complete", False),
        "api_success": sum(row.get("api", {}).get("status") == 200 for row in rows),
        "collection_seconds": distribution([row.get("collection_seconds") for row in rows]),
        "sample_gap_seconds": distribution([b["elapsed_seconds"]-a["elapsed_seconds"] for a, b in zip(rows, rows[1:])])}
    cameras = sorted({camera for row in rows for camera in row.get("cameras", {})})
    result["cameras"] = {}
    for camera in cameras:
        values = [row.get("cameras", {}).get(camera, {}) for row in rows]
        playlist = [value.get("playlist", {}).get("age_seconds") for value in values]
        checks = [value["clip_check"] for value in values if "clip_check" in value]
        starts = [value.get("recorder", {}).get("started") for value in values]
        starts = [value for value in starts if value is not None]
        result["cameras"][camera] = {
            "playlist_age_seconds": distribution(playlist),
            "playlist_stale_over_20s": sum(value is not None and value > 20 for value in playlist),
            "playlist_missing": sum(value is None for value in playlist),
            "missing_segment_samples": sum(value.get("playlist", {}).get("missing_segments", 0) > 0 for value in values),
            "vision_heartbeat_age_seconds": distribution([value.get("vision", {}).get("heartbeat_age_seconds") for value in values]),
            "recorder_heartbeat_age_seconds": distribution([value.get("recorder", {}).get("heartbeat_age_seconds") for value in values]),
            "observed_recorder_start_changes": sum(a != b for a, b in zip(starts, starts[1:])),
            "child_private_bytes": distribution([value.get("recorder", {}).get("child_memory", {}).get("private_bytes") for value in values]),
            "child_rss_bytes": distribution([value.get("recorder", {}).get("child_memory", {}).get("rss_bytes") for value in values]),
            "output_not_started_samples": sum(value.get("recorder", {}).get("output_started") is False for value in values),
            "observed_stop_reasons": dict(Counter(value.get("recorder", {}).get("stop_reason") for value in values if value.get("recorder", {}).get("stop_reason"))),
            "clip_checks": len(checks),
            "successful_decodes": sum(check.get("decode_exit") == 0 and check.get("stable_during_check") for check in checks),
            "full_clip_decode_mode": all(row.get("decode_seconds", 0) == 0 for row in rows),
            "failed_checks": sum(check.get("decode_exit") != 0 or not check.get("stable_during_check") for check in checks),
            "check_filenames_unique": len({check.get("file") for check in checks if check.get("file")}),
            "codec_combinations": dict(Counter("/".join(stream.get("codec_name", "?") for stream in check.get("probe", {}).get("streams", [])) for check in checks)),
        }
    result["units"] = {}
    units = sorted({unit for row in rows for unit in row.get("units", {}) if unit != "error"})
    for unit in units:
        values = [row.get("units", {}).get(unit, {}) for row in rows]
        events = [value.get("memory_events", {}) for value in values]
        result["units"][unit] = {
            "inactive_samples": sum(value.get("ActiveState") != "active" for value in values),
            "main_rss_kib": distribution([value.get("main_rss_kib") for value in values]),
            "memory_current_bytes": distribution([int(value["MemoryCurrent"]) for value in values if value.get("MemoryCurrent", "").isdigit()]),
            "main_pid_changes": sum(a.get("MainPID") != b.get("MainPID") for a, b in zip(values, values[1:])),
            "memory_events_delta": {key: int(events[-1].get(key, 0))-int(events[0].get(key, 0)) for key in ["max", "oom", "oom_kill"]},
        }
    result["catalog_heartbeat_age_seconds"] = distribution([row.get("catalog", {}).get("heartbeat_age_seconds") for row in rows])
    result["caveat"] = "Point samples and selected settled clips; not a client playback test or a complete decode of every recording. Interventions must be reported separately."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    output = json.dumps(summarize(rows), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output)
