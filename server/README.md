# Server

The server reads Tapo RTSP streams, samples frames, runs the trained YOLO model,
and sends Telegram alerts with snapshots.

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
already installed the cu128 torch build). It expects the model at:

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

### `/restart` Telegram command

Authenticated users can send `/restart` to gracefully stop camera workers,
release streams, and replace the current Python process with the same executable
and arguments. This is a real process restart, not only a rediscovery sweep.

### `/quality` Telegram command

`/quality stream1`, `/quality stream2`, and `/quality auto` control which Tapo
RTSP stream each camera consumes. The server starts on `stream1` by default.
`stream1` forces the high quality stream, `stream2` forces the lower bandwidth
stream, and `auto` starts conservatively on `stream2`, promotes stable cameras to
`stream1`, and falls back to `stream2` when FPS drops, frames stall, or
reconnects begin.

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
