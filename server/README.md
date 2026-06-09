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
- `TAPO_RSTP` (comma-separated RTSP URLs, ordered to match the cameras in
  `server/aviary_server/config.py`)
- `AVIARY_MODEL_PATH` if running outside Docker

## Docker Run

```bash
docker compose -f compose.dev.yml up --build
```

The Docker container expects the model at:

```text
server/models/current/bird_detector.pt
```

## Local Python Run

From the repo root (see the top-level README for installing uv):

```bash
uv sync
uv run server
```

On Apple Silicon, set the model `device` to `mps` in the `ModelConfig` defaults
in `server/aviary_server/config.py`.

## Camera Config

Cameras are hardcoded in `CAMERA_SPECS` in `server/aviary_server/config.py`. Each
entry maps positionally to a URL in the comma-separated `TAPO_RSTP` env var and
has:

- `name`: stable camera identifier.
- `enabled`: whether to start the stream.
- `sample_fps`: inference sampling rate.
- `reconnect_seconds`: delay after a stream failure.
- `alert_zones`: optional `ZoneConfig` polygons. If any are configured, only
  detections inside those zones send alerts.

The RTSP URL for each camera comes from the matching entry in `TAPO_RSTP`. If no
zones are configured, any visible bird can alert.
