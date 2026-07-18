# Server

The server reads Tapo RTSP streams, samples frames, runs the trained YOLO model,
and sends Telegram alerts with snapshots.

Outbound Telegram photos pass a privacy screen first: a stock COCO model checks
each image for people, and any frame showing a person is withheld — the alert or
report text still arrives, with a note that the photo was held back. Only
Telegram uploads are screened; local snapshots, collection and memory photos are
untouched. `PRIVACY_FILTER=0` disables it (see `.env.example` for the knobs).

## Setup

Copy and edit environment variables:

```bash
cp .env.example .env
```

Update:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_USER_IDS`
- `TAPO_CREDENTIALS` (`user:password` — only the Tapo camera-account
  credentials; cameras are auto-discovered on the LAN, see below)
- `TAPO_DISCOVERY_CIDR` (optional — override the scanned subnet when auto-detect
  picks the wrong interface)
- `MODEL_PATH` (comma-separated for multiple models — each runs a separate pass
  per frame and their detections are merged)

## Run

From the repo root (see the top-level README for installing uv):

```bash
uv run server
```

This runs the server natively in the uv venv on the Linux/RTX 5060 host (`uv sync`
already installed the cu128 torch build).

To run it in the background with boot autostart instead, use `uv run server-up`
(a systemd user service runs the dashboard inside a dedicated tmux session). Re-run
`uv run server-up` — or just `uv run server` — to attach to the live dashboard; Esc
or Ctrl-C detaches and leaves it running. `uv run server-down` stops it and removes
the autostart.

It expects the model at:

```text
data/server/models/current/object_detector.pt
```

## Camera Discovery

Cameras are not configured by hand. On startup the server scans the local subnet
for Tapo cameras and consumes whatever it finds:

1. Sweep every host on the subnet with a real RTSP `DESCRIBE` probe.
2. Use `TAPO_CREDENTIALS` to confirm the credentials are accepted and the stream path
   (`/stream1`) exists.
3. Build the full credentialed RTSP URL per confirmed camera and start consuming
   it. Each camera is named `camera-<host-ip>` for a stable identity across
   rediscovery.

Discovery deliberately avoids a separate throwaway TCP port probe. Each host gets
one RTSP handshake, with retries for transient drops, because small WiFi cameras
can intermittently miss when they are poked twice while already streaming.

The subnet is auto-detected from the host's primary IPv4 address (assumed `/24`),
or taken from `TAPO_DISCOVERY_CIDR` when set. The startup scan is non-fatal: if no
cameras are found the server logs a warning and keeps running so you can rerun
discovery once the cameras boot.

The RTSP port and stream path are pure code config, not env vars — see
`DiscoveryConfig` in `server/lib/config.py` (`rtsp_port`, `stream_path`, plus the
sweep concurrency and timeout tunables). Any visible configured object can alert.

### `/discover` Telegram command

Authenticated users (those in `TELEGRAM_USER_IDS`) can send `/discover` to the
bot to re-run the network scan at runtime. The bot acknowledges immediately, then
starts consuming any newly found cameras and replies with a summary (hosts
scanned, cameras added, auth failures). Cameras are de-duplicated by host, so
rerunning `/discover` only starts streams for cameras that are not already
active.

### `/watchlist` — choose which cameras stream (by MAC)

Every camera the sweep confirms with working credentials is cached by its MAC
address (read from the kernel ARP table, which the probe itself refreshes) in
`data/server/camera_registry.json`, together with its last-known IP. `/status`
and the `/discover` report show each camera's MAC next to its IP-derived name.

- `/watchlist` — lists every cached camera, grouped into *on the watchlist* and
  *discovered, not on the watchlist*, each with IP + MAC and live state.
- `/watchlist allow <MAC>` — adds a camera to the watchlist and starts its
  stream immediately (if it's offline, the monitor thread waits and connects
  the moment it appears).
- `/watchlist remove <MAC>` — removes it and stops the stream immediately.

An **empty** watchlist means no filtering — every discovered camera streams
(the out-of-the-box behavior). Once any MAC is listed, only listed cameras are
streamed; the rest are still cached and shown so they can be allowed later.

### `/restart` Telegram command

Authenticated users can send `/restart` to gracefully stop camera workers,
release streams, and replace the current Python process with the same executable
and arguments. This is a real process restart, not only a rediscovery sweep.

### `/detections` Telegram command

Every positive inference updates a daily JSON file under
`data/server/detection/YYYY-MM-DD.json`. The log stores merged visibility
intervals per camera and bird label, plus observation counts and max confidence,
so it stays compact while still answering "how long was this bird detected
today?" Send `/detections` for today's totals, `/detections percy` for one bird,
or `/detections percy 2026-06-27` for a specific UTC day.

### `/quality` Telegram command

`/quality stream1`, `/quality stream2`, and `/quality auto` control which Tapo
RTSP stream each camera consumes. The server starts on `stream1` by default.
`stream1` forces the high quality stream, `stream2` forces the lower bandwidth
stream, and `auto` starts conservatively on `stream2`, promotes stable cameras to
`stream1`, and falls back to `stream2` when FPS drops, frames stall, or
reconnects begin.

### `/flash`, night-find spotlights, and reboot-on-wedge (Tapo cloud control)

Tapo's ONVIF surface can't drive the cameras' built-in spotlights or reboot
them, so these features use the cameras' proprietary local HTTPS API (via
`pytapo`), which authenticates with the Tapo **cloud** account password — set
`TAPO_CLOUD_PASSWORD` in `.env`. Without it the features stay dormant (`/flash`
explains what's missing). See `server/lib/tapo.py`.

- `/flash` — per-camera spotlight status (device truth, with an IR marker).
- `/flash on | off | toggle [camera]` — force the spotlight; the camera can be
  an IP, a last-octet shorthand (`.19`), or a name fragment (`cockatiel`).
  Omitting the camera addresses every streaming camera.
- **Night finds light themselves**: a `/find` that starts while cameras sit in
  night/IR turns those spotlights on for the duration of the search and
  restores them after — a lamp the user forced via `/flash` is never touched.
  While a lamp is forced, that camera's IR flag is frozen so the lit,
  full-colour frames can't fake a "daylight" transition to the sleep tracker,
  auto-find, or the caretaker's night mode.
- **Reboot-on-wedge**: a camera whose stream stays unhealthy for ~3 minutes
  despite reconnects is power-cycled once via the API (at most once per
  10 minutes per camera), with a Telegram note. Cameras often wedge with RTSP
  dead but HTTPS alive, which is exactly what this repairs.

## Sleep tracking

The server watches the cameras' IR (night) signal to track the flock's sleep.
When the room goes fully dark (every camera in IR) a night opens; when it
lightens for good in the morning the night finalizes and is scored 0–100:

- **Duration** vs the 10–12h of darkness the birds need (steep penalty under 8h).
- **Consistency** — how close bedtime and wake-up are to their rolling usual.
- **Light at night** and **disturbances** (night-motion bursts, including
  possible cockatiel night-frights) dent the score.

Ask the bot **`/sleep`** for last night's score and summary, **`/sleep week`**
for the 7-night trend, or just "how did the birds sleep?". `/status` shows a
"night in progress" line while they're asleep. Set `SLEEP_MORNING_REPORT=1` to
also get a short summary each morning when they wake. Nightly records persist
under `data/server/sleep/` so the trend (and the consistency baseline) build up
over time. Sleep is measured at the room/flock level — individual birds can't be
told apart in the dark — so it's reported as "the birds' sleep".
