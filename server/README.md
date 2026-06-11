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
  `server/lib/config.py`)
- `AVIARY_MODEL_PATH` if running outside Docker

## Run

From the repo root (see the top-level README for installing uv):

```bash
./scripts/server.sh
```

This runs the server natively in the uv venv, installing the correct GPU torch
build for the machine first. It expects the model at:

```text
data/server/models/current/object_detector.pt
```

A `server/Dockerfile` is still provided if you'd rather containerize the server.

On Apple Silicon, set the model `device` to `mps` in the `ModelConfig` defaults
in `server/lib/config.py`.

## Camera Config

Cameras are hardcoded in `CAMERA_SPECS` in `server/lib/config.py`. Each
entry maps positionally to a URL in the comma-separated `TAPO_RSTP` env var and
has:

- `name`: stable camera identifier.
- `enabled`: whether to start the stream.
- `sample_fps`: inference sampling rate.
- `reconnect_seconds`: delay after a stream failure.

The RTSP URL for each camera comes from the matching entry in `TAPO_RSTP`. Any
visible configured object can alert.
