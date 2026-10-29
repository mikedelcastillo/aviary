# Aviary Observatory

A local, video-first companion to the existing Aviary collector. Open to a live
camera wall, swipe right for Reels, then swipe again for the chronological feed.
The website is a Next.js/TypeScript static export and an installable home-screen
PWA. A separate Android TV client uses the same authenticated LAN API.

This is a working development deployment, not a completed reliability release.
The sustained test has reproduced upstream camera timeouts and slow recorder
recovery. Automatic VLM publishing is disabled: evaluated models made unsupported
behavior claims. See [measured findings](../docs/OBSERVATORY-FINDINGS.md),
[tagging](../docs/DAILY-TAGGING.md), [camera control](../docs/TAPO-CONTROL.md),
[TV setup](../docs/ANDROID-TV.md), and
[movable-camera calibration](../docs/MOVABLE-CAMERA-CALIBRATION.md).
The [acceptance gates and next work](../docs/NEXT-STEPS.md) separate the remaining
reliability, curation and spatial-tracking work from the implemented foundation.
Zone drawing now uses an immutable screened picture with a short-lived owner
token; see [zone references](../docs/CAMERA-ZONES.md). The separate
`/api/readiness` route checks video outputs and returns 503 for stale cameras;
`/api/health` remains process liveness only.

## Processes and ownership

| Component | Responsibility |
| --- | --- |
| `api/` — Rust/Axum | Public HTTP entry point; sessions, indexed event queries, favorites, media ranges, HLS, static files; proxies other routes to the loopback adapter |
| `archive-core/` — Rust | Incremental import of local journal copies into SQLite/FTS |
| `backend/app.py` — Python | Camera/zone control, local model calls, archive photos, learning and clip workflows |
| go2rtc | One shared upstream RTSP connection per configured camera; loopback relay only |
| `backend/recorder.py` | Bounded FFmpeg decode, HLS, audio/video clips, latest inference frame, generated-file retention |
| `backend/worker.py` | Sampled person screening, bird detections/tracks, guarded 2D zone visits |
| `backend/catalog.py` | Index completed recordings independently of slow inference |
| `backend/curator.py` | Local visual review candidates and diverse learning samples |
| `backend/sync.py` | Optional read-only SSH journal pulls; writes local copies only |
| `web/` | Static Next.js UI; no Node server needed in production |

SQLite in WAL mode owns persistent state. Redis and PostgreSQL are not required
by this implementation; the measured indexed-query workload does not justify
adding them yet. Python remains for camera/model integration. The public API and
archive importer are Rust; this is a hybrid migration, not an all-Rust claim.

## Local installation

Use a separate development host. Never point writable runtime, cache, database,
model output, or recording paths at an original archive or production collector.
Keep deployment configuration, bird roster, network addresses, credentials,
camera images, recordings, APK test artifacts and reports outside Git.

Requirements: Linux, FFmpeg with the selected encoder, a current Rust toolchain,
Python 3.11+, Node 24+ for the web build, and a local go2rtc installation. The
optional vision environment additionally needs the project's compatible Torch,
Ultralytics, OpenCV, supervision and trackers dependencies; consult the recorded
versions in the findings and use a separate ML environment from the API adapter.
Ollama and downloaded model weights remain on local machines.

From `observatory/`:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd web
npm ci
npm run typecheck
npm run build
cd ..
cargo build --release --locked --manifest-path api/Cargo.toml
cargo build --release --locked --manifest-path archive-core/Cargo.toml
```

Create private `runtime/config.json` from `config.example.json`, replace the
example paths, and generate a random salt and a PBKDF2-SHA256 passcode hash with
600,000 iterations. Do not store the plaintext passcode in the configuration.
Set the file permissions to owner-only. Camera credentials belong in the private
go2rtc/ONVIF configuration, never in a frontend bundle.

```sh
export AVIARY_HOME_CONFIG=/absolute/private/runtime/config.json
.venv/bin/python -m backend.store /absolute/local/journal-copies
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8741 --no-access-log
# In a separate service, with observatory/ as its working directory:
AVIARY_LISTEN=127.0.0.1:8740 api/target/release/aviary-api
```

`AVIARY_WEB_ROOT` defaults to `web/out`; `AVIARY_ADAPTER_URL` defaults to the
loopback adapter on port 8741. Set `AVIARY_LISTEN` to a LAN interface only for
private-network clients such as the TV. Use private HTTPS for the home-screen
PWA; keep the relay, adapter and model server off public interfaces. Remote
access can use Tailscale Serve with Funnel disabled. On iPhone/iPad, open the
HTTPS address in Safari, unlock, then Share → Add to Home Screen.

For automatic TV discovery, run `.venv/bin/python -m backend.discovery` as a
separate service. It publishes `_aviary._tcp` on the LAN's selected IPv4
interface, with the app port and no credentials. The client validates the small
`/api/discovery` identity response before asking for the passcode. Optional
private configuration `discovery.address` selects an explicit LAN interface and
`discovery.port` overrides 8740. Discovery does not expose the relay or bypass
authentication; it requires multicast reachability on the local network.

Initialize before starting catalog/vision/curator. Run each worker as its own
service, with an explicit working directory and `AVIARY_HOME_CONFIG`. Workers
must have access only to their required local data. Keep `curation_auto_publish`
false until a representative, human-labeled evaluation passes.

## Verification

All examples below belong on a development host with fictional fixtures or
local copies. `pytest` is a development dependency.

```sh
.venv/bin/python -m pytest tests/test_home.py tests/test_catalog.py tests/test_learning.py tests/test_ptz.py
# In the ML environment:
python -m pytest tests/test_learning_sampling.py tests/test_learning_import.py tests/test_view_guard.py
AVIARY_RUST_BIN=/absolute/path/to/aviary-api .venv/bin/python -m pytest tests/test_rust_api.py
```

`tests/browser.mjs` uses an isolated Playwright installation and Google Chrome.
Provide `AVIARY_URL`, `AVIARY_PIN_FILE` and `AVIARY_SCREENSHOTS`; an optional
`PLAYWRIGHT_MODULE` can identify the installed module. Screenshot output is
private evidence. The smoke test deliberately records a failure if any live
camera stops advancing; an HTTP 200 response alone does not establish playback.
It also checks swipes, replay decoding, one selected live audio source, Feed
images, learning-frame loading, and phone/tablet bounds. Chrome emulation does
not replace physical Safari/iPad or TCL acceptance testing.

See `bench/` for repeatable API and sustained-pipeline measurements. Preserve
the raw private results and report duration, interruptions, fixture scope and
measurement limits alongside any performance claim.

## Boundaries

- Live video and original recordings are unfiltered. Sampled person screening
  is not a guarantee that an entire clip contains no person.
- Detector identity is an estimate. Proximity to a feeder/water zone is not
  proof of eating/drinking, and missing observations are not proof of absence.
- Current zone tracking is 2D and invalidates moved views. Metric 3D trajectories
  require printed-marker validation, camera intrinsics and synchronized views.
- Reels supports selected clips, recent recordings, audio, favorites and view
  tracking. Automatic unseen-clip montage editing remains future work.
- Reviewed learning exports preserve the original annotator's label indices;
  uncertain answers and skips are excluded from training.
