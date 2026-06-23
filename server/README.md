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

1. Sweep every host on the subnet for an open RTSP port (`:554`).
2. For each reachable host, perform an RTSP `DESCRIBE` handshake using
   `TAPO_CREDENTIALS` to confirm the credentials are accepted and the stream path
   (`/stream1`) exists.
3. Build the full credentialed RTSP URL per confirmed camera and start consuming
   it. Each camera is named `camera-<host-ip>` for a stable identity across
   rediscovery.

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
