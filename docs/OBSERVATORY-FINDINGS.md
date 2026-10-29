# Observatory: audit and measured findings

Status: development deployment, 2026-10-29. This document contains sanitized
engineering findings only. Deployment addresses, people, bird identities,
credentials, images, and individual observation records are deliberately omitted.

## Boundaries

The existing production installation and archive are read-only sources. All new
services, benchmarks, generated media, and tests run on a separate development
host. Inference and media remain local. Runtime configuration and reports with
private deployment details are excluded from version control.

## Existing application

- Capture and inference share a synchronous path. Full-resolution decoding and
  automatic FFmpeg thread counts increase resident memory for each camera.
- Observation interval updates repeatedly sum prior intervals. Incremental
  accounting removes this cost while preserving duration totals.
- Camera discovery opens additional streams; periodic PTZ resets can conflict
  with a person using a new live control panel. The new application does not
  modify these production jobs.
- Archive records include structured detections and generated prose. Generated
  captions have contained unsupported identities and behavior claims. Identity
  must come from structured classifier evidence, with uncertainty preserved.
- Existing collection metadata serialized a camera object containing its RTSP
  URL. The code change prevents future URL serialization; it does not rewrite
  or delete existing archive records.
- Historical inventory is overwhelmingly photographs and JSON, not video.
  A video-first feed requires recording new footage; photographs cannot recover
  the missing movement, audio, or exact event sequence.

## Component benchmarks

Measurements below were made on the development host, using the same recorded
input for each comparison. They are not an end-to-end application speedup.

| Component | Before | After | Qualification |
|---|---:|---:|---|
| 5,000 duration updates with pre-existing intervals | 0.994 s | 0.220 s | Equal final duration and observation count; 4.51x faster |
| Three simultaneous full-resolution decoders, default vs two threads each | 616.93 MiB peak | 239.23 MiB peak | About 61% less memory; maximum offline throughput decreased |
| Decoder CPU time for the same workload | 9.422 s | 4.613 s | About 51% less CPU time |
| Latest 50 archive records | 653.9 ms scan/parse | 0.0331 ms indexed query | Same IDs; warm indexed-query microbenchmark |
| Full archive ingestion, Python | 12.529 / 12.503 s | — | Two runs; 26,052–27,000 KiB peak RSS |
| Full archive ingestion, Rust | — | 6.824 / 6.557 s | Two runs; 8,640–8,928 KiB peak RSS; matching canonical output |
| 20 s video encoding, CPU x264 | 4.008 s | — | 143,044 KiB peak, 2.25 MB output |
| 20 s video encoding, GPU NVENC | — | 2.474 s | 234,088 KiB peak, 3.63 MB output; different rate control |

The native HTTP comparison used one isolated 90,575-record snapshot, warm
loopback requests, and matching canonical JSON. These results are separate
from the earlier archive-import snapshot:

| HTTP measurement | Python | Rust | Qualification |
|---|---:|---:|---|
| Serial event requests, median / p95 | 5.325 / 7.036 ms | 2.939 / 3.630 ms | 200 requests |
| Concurrency 16 throughput | 170.4 requests/s | 244.6 requests/s | 800 requests |
| Concurrency 16 p95 | 140.257 ms | 142.272 ms | Tail latency did not improve |
| One-MiB media ranges, median / p95 | 11.137 / 13.821 ms | 5.484 / 9.946 ms | 80 exact-length HTTP 206 responses |
| HTTP process peak RSS | 64,748 KiB | 11,800 KiB | Excludes the separate Python adapter and other workers |

The approximately 82% reduction in HTTP-process peak RSS is not a total-stack
memory reduction. The hybrid deployment still runs Python adapters and local
models. The benchmark does not include TLS, an external network, or browser
rendering. See `observatory/bench/benchmark_http.py` for the method.

Rust and Python importers produced 89,412 unique records from 89,413 imports.
The canonical result digest matched. This supports adopting the Rust importer;
it does not establish that every Python component should be rewritten.

Detector evaluation used 174 existing test images with 362 annotations. Their
independence from prior training has not been established. At confidence 0.4,
640/960/1280 pixel inputs gave recall 0.619/0.666/0.707 and median latency
47.12/84.26/147.33 ms respectively. Reducing resolution saves compute but loses
small or difficult birds. The current live worker uses 960 pixels.

A synthetic six-track benchmark had no identity switches in separated paths.
This is a plumbing check, not evidence of reliable identity across real cage
occlusion, similar birds, PTZ changes, or infrared mode. Real annotated video
evaluation remains necessary.

## Streaming and audio findings

The application uses one shared camera connection per relay source. A single
decode/encode process provides a regular H.264 timeline to HLS and rolling MP4
segments, plus a replaceable JPEG for inference. Browsers do not open their own
camera RTSP connections. Inference cannot build an unbounded frame queue.

Observed source timestamp discontinuities caused unusable segment boundaries.
Wall-clock input timestamps and a normalized 15 fps output were introduced.
These mitigate observed timestamp behavior; they do not prove all remaining
connection problems originate in camera firmware.

A recorder restart exposed an application bug: FFmpeg's latest-JPEG output
required explicit overwrite permission even with `image2 -update 1`. Its parent
service remained active while child processes repeatedly exited. The fix adds
`-y`, bounded error diagnostics, and a watchdog based on actual playlist age.
Health must report output freshness, not only process state. Interrupted old
MP4s can lack a final `moov` atom; they must not enter the playable feed.

All three current RTSP sources expose mono G.711 A-law at 8 kHz. The browser
pipeline converts this to AAC. Upsampling to 16 kHz does not restore frequencies
above the source's 4 kHz Nyquist limit. Bird-call models must be evaluated on
this actual degraded bandwidth before their species labels are trusted.

After the restart fix, probes confirmed H.264 at 15 fps plus AAC in new clips
from all three cameras. All three browser players reached ready state 4 and
played. The observation run included deployments and a recording outage; it
must not be reported as an uninterrupted, passed stability test.

A later read-only inventory inspected 1,754 settled recording filenames over
roughly three hours of retained footage. There were 1,698 successful probes,
all H.264 at 15 fps with AAC, and 57 sampled full video/audio decodes with zero
failures. Rolling retention removed 56 files before inspection; these are
unavailable evidence, not corrupt media. Filename-plus-duration coverage still
showed approximately 33-second and 109-second gaps corresponding to reconnects.
This estimate cannot detect missing source frames concealed by duplicated output.

The first run collected 481 samples over approximately two hours. Each camera
had 45 samples with playlist age above 20 seconds, reaching roughly 690 seconds.
After the overwrite fix, 288 samples covered another 72 minutes: 287 contained
camera data and one failed at the API check. No sampled playlist exceeded 20
seconds; maximum ages were 9.56, 12.21, and 4.82 seconds. Fifteen duration probes
per camera were approximately 20.015-20.047 seconds. Duration probes are not
full decodes or proof of continuous playback.

Later logs still showed recorder-child restarts on two cameras and relay-side
upstream timeouts on all three. Malformed RTP/RTSP input also occurred. This
evidence does not isolate camera firmware, Wi-Fi, connection limits, or relay
handling as the sole cause. Existing production camera connections remain in
addition to the new development relay sources.

A planned three-hour independent observation began at 2026-10-29 02:44:32 UTC.
Before the corrective recorder deployment, 226 samples covered 56.25 minutes:
all API health requests succeeded, but camera playlist ages reached
16.67/22.21/99.56 seconds and observed child-start changes were 2/1/21.
All 33 sampled full clip decodes passed. The run was deliberately truncated for
remediation and did not complete its planned duration. Raw data and the reason
remain private. `observe_pipeline.py` and `summarize_pipeline.py` document the
method. Sampled server liveness is not a long-running browser test.

The recorder's initial 512 MiB cgroup limit produced reclaim churn. In a
30-second sample there were 246 new limit events, approximately 42.5 MiB of
directly reclaimed pages, and 26 major faults, but no OOM kills and only about
0.003% memory full-stall time. At 02:49:49 UTC the limit was raised to 1 GiB
without restarting the recorder, explicitly marking an observation intervention.
Over the next three minutes there were no further limit events or direct-reclaim
pages. Charged memory increased from 511.8 to 664.7 MiB, almost entirely file
cache (+147.0 MiB versus +1.33 MiB anonymous memory). This removes needless
short-window reclaim; it does not prove a source-dropout fix. RSS sums can count
shared pages repeatedly, and cgroup memory includes file cache and kernel usage.
During a later source-reconnection attempt, however, charged anonymous memory
briefly reached approximately 943 MiB before the startup watchdog killed the
hung child. That recovery spike was not file cache.

Three independently limited recorder services and a separate retention service
replaced the combined recorder at 03:40:59 UTC. Input probing and mux queues are
bounded, and supervision uses actual new output plus per-child anonymous memory
and swap. Eight fault tests passed. An offline paused-encoder experiment stopped
the child in 2.461 seconds and produced fresh replacement HLS after 2.710 seconds,
while a healthy sibling finalized its recording. That fixture used shortened
two-second startup and 0.4-second shutdown limits; the deployed defaults are
15-second startup, 12-second output idle, and three-second graceful shutdown.

A fresh two-hour server observation began at 03:42:40 UTC and is in progress.
Initial post-deployment clips decoded cleanly, but natural source/recovery
interruptions continue; isolation must not be described as uninterrupted video.
Current complete HLS windows can be valid while restart boundaries are wrong.
A captured natural restart reset video PTS from 385.997333 to 0.064 seconds with
no discontinuity tag or sequence in the new playlist. This is direct evidence
for fixing restart signaling, not proof that it explains every client failure.
The HLS protocol requires a discontinuity marker when the timestamp sequence
changes. [RFC 8216 section 4.3.2.3](https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.3).

The publisher correction separates encoded epochs from stable public media
identity and signals timestamp discontinuities. Eighteen recorder unit/fault
checks passed. With the same three HLS.js player instances, an offline synthetic
video/audio benchmark survived graceful restart, SIGKILL, and a 23-second
outage across 127 manifest updates, with no fatal or ended events and one
nonfatal long-outage stall. This does not validate actual camera-to-screen
latency or Android playback. Sequential deployment at 04:17:21-04:18:05 UTC
produced fresh manifests on the first two views; the third service remained
without output at that cutoff. No source/relay settings changed. The ongoing
observations include that intervention and cannot be called an uninterrupted
stability pass.

The already-playing browser observer stopped returning samples at 04:17:28 UTC
during that migration while its renderer main thread consumed about one CPU
core. The initial offline test began on the new publisher and omitted migration
from old sequence numbers. An offline comparison reproduced failed playback
on a huge sequence jump; contiguous low and already-high ranges both advanced
normally. The exact live main-thread stack was not captured. A correction now
persists a separate atomic continuity checkpoint and recovers sequence state
from valid playlists or existing segment IDs rather than changing ranges.
Twenty-one recorder checks passed. Following the 04:32:12-04:32:13 UTC restart,
all three replacement children had fresh output/checkpoints at 04:33:37 UTC,
with bounded sequence advances and one discontinuity increment each. This does
not repair an already-frozen client or establish source continuity/latency.
The future browser harness now bounds renderer evaluations and writes a
two-second heartbeat from the separate Node process. Two isolated real-Chrome
tests verified timeout detection, continued heartbeat during a renderer loop,
and failed-result persistence by the complete observer. The original frozen
session's raw data and CPU diagnostic were preserved before its authorized
termination at 04:36:20 UTC. Changing the harness does not repair that run.

At the report cutoff of 04:35:10 UTC, the ongoing server observation held 211
samples over 52.5 minutes. All liveness requests succeeded, but playlist ages
reached 42.14/42.78/360.82 seconds; thirty clip checks included three unreadable
containers and one four-frame video-only clip. All four were independently
verified excluded from the playable catalog. The browser's actual last sample
was at 04:17:28 UTC after 37.75 minutes, before its renderer froze. Neither run
is a passed two-hour stability test. A guarded ten-minute browser follow-up
completed from 04:36:25 to 04:46:25 UTC: 41 samples, no hang or
page/console/request errors, no sampled interval without frame progress, and
0/1/1 observed playback resets. This is recovery evidence, not uninterrupted
continuity. After it closed, one guarded two-hour follow-up began at 04:47:00
UTC, scheduled through 06:47:00 UTC.

Fresh file timestamps also do not establish fresh visual content. Readable
camera overlays in three newly written capture images were roughly 12-16
seconds behind server time while file ages were only 0.5-1.3 seconds. Camera
clock offsets measured independently were about one second. Source, relay,
recorder, and client delay need separate measurements.

A bounded twenty-second sustained consumer of the existing loopback relay
preserved input RTP timestamps. One source's readable overlay remained about
21 seconds behind its image write time at the end, versus about 24 seconds at
the first image. Its contemporaneous recorder image was also about 24 seconds
behind despite a 0.2-second file age. This delay persisted beyond the first
cached frame. The other two bounded relay attempts failed with a demux timeout
and invalid input. No extra upstream camera connection was created. These
measurements do not isolate the camera, transport, or relay as the sole cause.

## Local models and editorial reliability

Larger local text models still made factual mistakes in small evidence tasks.
Counts, dates, elapsed time, and source selection therefore come from indexed
records. A local model may explain supplied facts but cannot create new facts.

The current visual judge examines three ordered video frames. Real judgments
included a perch incorrectly categorized as a feeder, with high self-reported
confidence. Automatic highlight publication remains disabled. Sampled frames
can miss brief actions and people between samples; this is not full-video
privacy screening. The owner review queue and original recordings are clearly
distinguished from selected highlights.

The separate offline audio prototype found 15 listening candidates totaling
16.548 seconds in nine clips containing 179.840 seconds of audio; feature
analysis and hashing took 2.688 seconds. Nine independent three-second audit
snippets preserve some negative-candidate evidence. A pinned BirdNET v2.4 trial
ran nine windows on one CPU thread in 0.608 seconds, excluding load and decode.
It produced unvalidated species scores, including a high-scoring unexpected
species. No human audio ground truth exists for this convenience sample, so
species precision, recall, singing detection, and individual identity remain
unmeasured. Neither the heuristic nor model is connected to the live feed or
training export. See `docs/BIRD-AUDIO.md` for method and limitations.

## Current stack and remaining work

The deployed frontend is now Next.js 16.4.3 with strict TypeScript throughout
the app components, built as a static export. No additional Node runtime is
needed to serve Aviary. The production build, existing API tests and browser
hydration check passed. Native Safari HLS avoids loading the JavaScript HLS
bundle, and only one camera's Listen control stays unmuted at a time. Chromium
now prefers HLS.js/MSE: in a concurrent ten-minute canary it had zero resets or
media errors, while native Chrome HLS had two media-error/reset events. The two
contexts loaded different UI builds, so this is not a controlled same-build
comparison and does not isolate the playback engine as the cause. A fresh
two-hour unmodified-context MSE browser observation is running. This is not a
physical iPad/Safari test, and server/source interruptions still affect clients.

Rust archive ingestion and the public HTTP service are deployed. Native routes
cover sessions/authentication, archive reads, selected zone operations, HLS,
video byte ranges, and the allowed playback WebSocket path. A loopback-only
Python adapter handles camera/model calls and the remaining API routes. Five
Rust integration tests covered native route behavior, binary WebSocket
forwarding, cross-origin authentication, path boundaries, and first-seen logic.
The initial HTTP rollout was rolled back after a database-lock startup timeout;
separating catalog work from the visual judge and removing external media probes
from write transactions enabled the subsequent deployment. SQLite WAL plus FTS
handles the measured archive queries; PostgreSQL and Redis have not demonstrated
a need in this single-household deployment.

Remaining acceptance work includes multi-hour uninterrupted playback and audio
checks, six-camera capacity testing, real clip judging evaluation, unseen-only
reel editing, bird-call evaluation, privacy evaluation, exact-model Tapo control
compatibility, printed-marker validation, and calibrated multi-view tracking.

## Primary references

- [go2rtc](https://github.com/AlexxIT/go2rtc): local stream sharing and browser transports.
- [TP-Link RTSP/ONVIF guide](https://www.tp-link.com/us/support/faq/2680/): official camera stream interfaces.
- [tolpo](https://github.com/signerless/tolpo): direct local Tapo protocols and control; MIT license.
- [3D-MuPPET](https://github.com/alexhang212/3D-MuPPET): calibrated multi-view pigeon pose research, not a drop-in pet-bird identity model.
- [SLEAP](https://sleap.ai/): trainable multi-animal pose estimation.
- [Anipose](https://anipose.readthedocs.io/en/latest/start3d.html): calibration and multi-camera reconstruction.
- [BirdNET Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer): candidate local audio analysis to evaluate against camera bandwidth.
