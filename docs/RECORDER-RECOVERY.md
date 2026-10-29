# Bounded recorder recovery

The recorder supervises each FFmpeg process independently. Startup must produce
a newly written HLS playlist within 15 seconds; an existing playlist from the
previous process does not count. Once output starts, twelve seconds without a
playlist update stops that child. Deadlines use a monotonic clock, so adjusting
the wall clock cannot extend a stalled recording indefinitely.

Input discovery is limited to 1 MiB and two seconds of analysis. Filter graphs
use two threads, rather than the host's entire CPU count. The output queue
waiting for optional audio is limited to 128 packets after 1 MiB of buffered
data. These are supported by FFmpeg 6.1. `rtbufsize` is also set to 4 MiB, but
it is format-specific and is not a global process memory limit. FFmpeg 6.1 does
not support the newer `filter_buffered_frames` option.

Every 0.5 seconds the supervisor samples that child's `/proc/PID/status`.
The graceful memory threshold is 384 MiB of anonymous resident memory plus
swap. Shared file mappings are reported as part of RSS but excluded from this
threshold because GPU driver mappings can otherwise make healthy children
appear oversized. At the threshold, the child receives SIGINT so FFmpeg can
finish its current MP4. After three seconds it receives SIGKILL. Reaching
512 MiB of private memory skips the grace period; memory is sampled during
grace as well. A process that cannot be reaped after SIGKILL is not replaced
with another decoder.

Each child starts a new process group. Recovery signals target only that
group, never the recorder's siblings or the relay. A hardware encoder stopped
for memory pressure switches to software encoding for ten minutes. Ordinary
repeated failures still fall back after three attempts. Heartbeats retain the
existing fields and now add child PID, memory, progress, fresh-output state,
output-idle time and stop reason. SIGINT/SIGTERM to the supervisor finishes
children through the same bounded shutdown path.

Private configuration can override the policy:

```json
{
  "recorder_limits": {
    "startup_seconds": 15,
    "stale_seconds": 12,
    "poll_seconds": 0.5,
    "shutdown_seconds": 3,
    "private_bytes": 402653184,
    "emergency_bytes": 536870912
  }
}
```

These checks are a sampled guard, not a hard allocation ceiling. Very fast
allocation can exceed a threshold between polls. Separate systemd cgroups
provide the final containment boundary. In the legacy shared service, the
parent cgroup's memory limit still applies to all cameras together.

## Per-camera service deployment

The new command supports `python3 -m backend.recorder --camera CAMERA_ID`.
That mode records only the named configured camera and does not run retention.
`python3 -m backend.recorder --retention-only` runs retention without cameras.
Running without either option preserves the existing combined service.

An operator can adapt this user-unit template to the deployment's real paths:

```ini
[Unit]
Description=Aviary recorder %i
After=network-online.target

[Service]
WorkingDirectory=/absolute/path/to/observatory
Environment=AVIARY_HOME_CONFIG=/private/path/to/config.json
ExecStart=/absolute/path/to/python -m backend.recorder --camera %i
Restart=on-failure
RestartSec=3
KillMode=control-group
TimeoutStopSec=8
MemoryAccounting=yes
MemoryHigh=640M
MemoryMax=768M
MemorySwapMax=128M
OOMPolicy=stop

[Install]
WantedBy=default.target
```

Use one instantiated unit per camera and one separate retention unit. Set
resource limits from measured normal and recovery usage, with enough total
headroom for the configured camera count. MemoryHigh is a reclaim/throttling
threshold and MemoryMax is the final hard limit. These are example values,
not a change to any running deployment.

Stop the combined recorder before starting the per-camera units, and do not
run two writers for one camera. Keep the current service definition available
for rollback. Check fresh HLS, complete MP4s, audio and inference JPEGs for
every camera after switching. Relay and observer services need no changes.

## Verification

All executable validation runs on the designated Linux build host:

```sh
cd observatory
python3 -m unittest discover -s tests -p test_recorder_health.py -v
python3 bench/benchmark_recorder_recovery.py \
  --fixture /private/path/to/offline-video.mp4 \
  --output /private/empty/audit-directory
```

The eight tests cover old-playlist startup, monotonic output stalling, memory
thresholds, command bounds, graceful MP4-style cleanup, a child that ignores
SIGINT, and an actual growing-memory child. They verify that a healthy sibling
survives both forced stops.

The offline benchmark starts two real software FFmpeg encoders. It freezes
one child, lets the supervisor terminate only that child, starts a replacement,
and verifies fresh HLS, uninterrupted sibling output and finalized readable
MP4s. Its output directory holds errors and structured memory/deadline traces.
It reads an existing file and does not connect to cameras, use an NVENC session
or change a service. The forcibly killed child's unfinished MP4 can remain
unreadable; catalog validation already excludes incomplete files.

Hardware encoder recovery and per-camera cgroup limits still require a
coordinated deployment test. Offline software results do not establish the
cause of a particular NVENC stall or its future memory use.

References: [FFmpeg options](https://www.ffmpeg.org/ffmpeg.html) and
[systemd memory controls](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html).
