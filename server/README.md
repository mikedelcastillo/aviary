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
- `TAPO_*_RTSP`
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
uv run server --config server/config/cameras.yaml
```

On Apple Silicon, set `model.device: mps` in `server/config/cameras.yaml`.

## Camera Config

Each camera has:

- `name`: stable camera identifier.
- `enabled`: whether to start the stream.
- `rtsp_url`: RTSP URL, usually from `.env`.
- `sample_fps`: inference sampling rate.
- `reconnect_seconds`: delay after a stream failure.
- `alert_zones`: optional polygons. If any are configured, only detections
  inside those zones send alerts.

If no zones are configured, any visible bird can alert.
