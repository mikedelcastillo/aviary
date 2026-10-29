"""Summarize observe_browser.mjs without treating reset intervals as continuity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(rows):
    snapshots = [row for row in rows if "videos" in row]
    if not snapshots:
        return {"complete": False, "samples": 0, "fatal_error": rows[-1].get("fatal_error") if rows else "No samples"}
    eligible = [row for row in snapshots if row["elapsed_seconds"] >= 30]
    output = {"samples": len(snapshots), "start": snapshots[0]["timestamp"], "end": snapshots[-1]["timestamp"],
        "elapsed_seconds": snapshots[-1]["elapsed_seconds"], "complete": rows[-1].get("complete", False),
        "fatal_error": rows[-1].get("fatal_error"),
        "fatal_operation": rows[-1].get("operation"),
        "expected_build": snapshots[0]["expected_build"], "eligible_samples_after_30s": len(eligible),
        "observed_version_at_start": snapshots[0].get("observed_version_at_start"),
        "forced_hlsjs": snapshots[0].get("forced_hlsjs", False),
        "page_errors": max(row["pageErrors"] for row in snapshots),
        "console_errors": max(row["consoleErrors"] for row in snapshots),
        "failed_requests": max(row["failedRequests"] for row in snapshots),
        "hidden_samples": sum(row["hidden"] for row in snapshots),
        "wrong_video_count_samples": sum(len(row["videos"]) != 3 for row in snapshots),
        "all_three_advance_without_reset": sum(len(row["videos"]) == 3 and all(
            video["advanced"] and not video["observedReset"] and video["ready"] >= 2 and not video["paused"] and not video["ended"] and not video["error"]
            for video in row["videos"]) for row in eligible), "slots": {}}
    for slot in sorted({video["slot"] for row in snapshots for video in row["videos"]}):
        values = [video for row in snapshots for video in row["videos"] if video["slot"] == slot]
        sample_values = [video for row in eligible for video in row["videos"] if video["slot"] == slot]
        events = defaultdict(dict)
        for video in values:
            for event, count in video["eventCounts"].items():
                events[video["elementId"]][event] = max(count, events[video["elementId"]].get(event, 0))
        totals = defaultdict(int)
        for counts in events.values():
            for event, count in counts.items():
                totals[event] += count
        longest = consecutive = 0
        for video in sample_values:
            consecutive = consecutive + 1 if video["frameDelta"] == 0 else 0
            longest = max(longest, consecutive)
        output["slots"][str(slot)] = {
            "samples": len(sample_values), "observed_resets": sum(video["observedReset"] for video in sample_values),
            "no_decoded_frame_progress_samples": sum(video["frameDelta"] == 0 for video in sample_values),
            "longest_consecutive_no_progress_samples": longest,
            "paused_samples": sum(video["paused"] for video in sample_values),
            "ended_samples": sum(video["ended"] for video in sample_values),
            "not_ready_samples": sum(video["ready"] < 2 for video in sample_values),
            "media_error_samples": sum(bool(video["error"]) for video in sample_values),
            "event_totals": dict(totals),
        }
    output["caveat"] = "Headless Chromium at a phone-sized viewport; no physical iPad/Safari, audio listening, background lifecycle, or pixel-content freeze test. Samples cannot prove no shorter inter-sample stalls."
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    result = json.dumps(summarize(rows), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(result)
    print(result)
