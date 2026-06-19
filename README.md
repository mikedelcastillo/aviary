# Aviary

Computer vision platform for monitoring a bird room with Tapo cameras.

The project is split into three working areas:

- `annotation/`: the `/annotation` web tool to collect images and label birds
  with bounding boxes (run `./scripts/annotation.sh`).
- `training/`: prepare datasets and train the Ultralytics YOLO detectors.
- `server/`: consume camera streams, run inference, and send Telegram alerts.

The first production goal is to identify each of the six birds. The model uses
one detection class per bird plus `unknown_bird` for visible birds that are not
confidently identifiable.

## Tooling

The whole repo is one [uv](https://docs.astral.sh/uv/) project. Install uv once
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then from the repo root:

```bash
uv sync                  # create .venv and install every subsystem's deps
```

Run any subsystem with a short named command (think `npm run`):

| Command | Runs |
| --- | --- |
| `uv run prepare-dataset` | normalize a YOLO label export into a training dataset |
| `uv run train` | train the Ultralytics YOLO detector |
| `uv run evaluate` | evaluate a trained model |
| `uv run extract-frames` | extract frames from an RTSP stream or video |
| `uv run server` | run the camera inference + alert server |

Pass flags after the command, e.g. `uv run extract-frames --help`.
Run from the repo root so the scripts' default relative paths resolve.

## Quick Start

1. Copy the environment template and fill in secrets:

   ```bash
   cp .env.example .env
   ```

2. Collect seed images:

   - Put phone photos in `data/annotation/raw/phone/`.
   - Extract Tapo day frames into `data/annotation/raw/tapo/day/`.
   - Extract Tapo infrared frames into `data/annotation/raw/tapo/ir/`.

3. Label images with the `/annotation` web tool — run `./scripts/annotation.sh`
   and open http://0.0.0.0:5000. It labels against `training/roster.yaml` (the
   single label roster for both the live and archive models — see
   `training/STRATEGY.md`) and writes YOLO `.txt` labels back next to each image.

4. Build a model end-to-end — prepare its dataset from the labeled raw images,
   train, and export the weights into `training/models/`:

   ```bash
   ./scripts/train_live.sh      # -> training/models/live.pt    (real-time CCTV detector)
   ./scripts/train_archive.sh   # -> training/models/archive.pt (photo-library catalog)
   ```

   Flags pass through to training, e.g. `./scripts/train_live.sh --epochs 200
   --device cuda:0`. To run the prepare/train/evaluate steps by hand instead, see
   [`training/README.md`](training/README.md).

5. Set `TAPO_CREDENTIALS=user:password` in `.env` (only the Tapo camera-account
   credentials), then run the server:

   ```bash
   ./scripts/server.sh
   ```

   Cameras are auto-discovered: the server scans the local subnet for hosts with
   RTSP `:554` open and a working `/stream1`, then consumes each one. Set
   `TAPO_DISCOVERY_CIDR` if auto-detect picks the wrong subnet (e.g. in Docker).
   Send the bot `/discover` to re-run the scan and pick up cameras at runtime. The
   scan port and stream path live in `DiscoveryConfig` in `server/lib/config.py`.

## Practical Dataset Targets

Start with at least 100 to 200 labeled boxes per bird. Include day mode,
infrared mode, cage bars, floor, perches, partial occlusion, close-up shots,
far camera views, and frames with multiple birds.

Phone photos are useful for bootstrapping identities. Camera frames are more
important for validation because they match the deployment view.

## Runtime Notes

`./scripts/server.sh` runs the server natively in the uv venv (it picks the
right GPU torch build via `scripts/install-gpu.sh`) and is the default path on a
Linux GPU machine. A `server/Dockerfile` is still provided if you prefer to
containerize. On Apple Silicon, Docker usually will not expose MPS acceleration
to PyTorch, so run natively and set the model `device` to `mps` in
`server/lib/config.py`.

## External References

- Ultralytics YOLO dataset / label format: https://docs.ultralytics.com/datasets/detect/
- Ultralytics detection training: https://docs.ultralytics.com/tasks/detect/
- Tapo RTSP/ONVIF guidance: https://www.tapo.com/faq/34/
