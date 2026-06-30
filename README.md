# Aviary

Computer vision platform for monitoring a bird room with Tapo cameras.

The project is split into three working areas:

- `annotation/`: the `/annotation` web tool to collect images and label birds
  with bounding boxes (run `uv run annotation` — a Next.js app).
- `training/`: prepare datasets and train the Ultralytics YOLO detectors.
- `server/`: consume camera streams, run inference, and send Telegram alerts.

The first production goal is to identify each of the six birds. The model uses
one detection class per bird plus `unknown_bird` for visible birds that are not
confidently identifiable.

## Platform assumption

This project runs on a single **Linux machine with an NVIDIA RTX 5060** (Blackwell,
CUDA **cu128**). `torch`/`torchvision` are pinned to the cu128 wheel index in
`pyproject.toml`, so `uv sync` installs the right GPU build with no post-step — there
is no install-gpu script and no Windows/ROCm/multi-GPU/Docker tooling.

## Tooling

The whole repo is one [uv](https://docs.astral.sh/uv/) project. Install uv once
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then from the repo root:

```bash
uv sync                  # create .venv, install every subsystem's deps + cu128 torch
```

Run any subsystem with a short named command (think `npm run`):

| Command | Runs |
| --- | --- |
| `uv run prepare-dataset` | normalize a YOLO label export into a training dataset |
| `uv run train` | train the Ultralytics YOLO detector (low-level) |
| `uv run train-live` | build the live CCTV detector end-to-end -> `data/models/live-NNN.pt` |
| `uv run train-archive` | build the archive catalog detector end-to-end -> `data/models/archive-NNN.pt` |
| `uv run evaluate` | evaluate a trained model |
| `uv run benchmark` | score every `data/models/*.pt` (held-out `test` split by default) |
| `uv run suggest` | propose boxes + roster labels for annotation images |
| `uv run import-collect-birds` | import server-collected frames into the annotation tree |
| `uv run import-immich-albums` | download Immich "Birds" album photos into the annotation tree |
| `uv run extract-frames` | extract frames from an RTSP stream or video |
| `uv run synth-ir` | synthesize IR-style frames from day frames |
| `uv run cut-paste-ir` | cut-paste IR augmentation |
| `uv run server` | run the camera inference + alert server in the foreground (attaches to the background server if one is already up) |
| `uv run server-up` | run the server in the background and start it on boot; re-run to attach to the live dashboard (Esc / Ctrl-C detaches) |
| `uv run server-down` | stop the background server and remove the boot autostart |
| `uv run tests` | run the pytest suite |

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

3. Label images with the `/annotation` web tool — run `uv run annotation` (Next.js)
   and open http://0.0.0.0:5000. It labels against `training/roster.yaml` (the
   single label roster for both the live and archive models — see
   `training/STRATEGY.md`) and writes YOLO `.txt` labels back next to each image.

4. Build a model end-to-end — prepare its dataset from the labeled raw images,
   train, and export versioned weights into `data/models/`:

   ```bash
   uv run train-live      # -> data/models/live-NNN.pt    (real-time CCTV detector)
   uv run train-archive   # -> data/models/archive-NNN.pt (photo-library catalog)
   ```

   Each run writes the next zero-padded sequence number, so previous models are
   preserved and never clobber the one the server may be loading. Flags pass
   through to training, e.g. `uv run train-live --epochs 200 --device cuda:0`. To
   run the prepare/train/evaluate steps by hand instead, see
   [`training/README.md`](training/README.md).

5. Set `TAPO_CREDENTIALS=user:password` in `.env` (only the Tapo camera-account
   credentials), then run the server:

   ```bash
   uv run server
   ```

   Cameras are auto-discovered: the server scans the local subnet for hosts with
   a working RTSP stream, then consumes each one. Set
   `TAPO_DISCOVERY_CIDR` if auto-detect picks the wrong subnet. Send the bot
   `/discover` to re-run the scan and pick up cameras at runtime. The scan port,
   stream paths, and timeout knobs live in `DiscoveryConfig` and the quality
   controller code under `server/lib/`.

   Bot commands include `/status` (runtime health), `/discover` (re-scan the
   LAN), `/restart` (re-exec the server process),
   `/detections [bird] [YYYY-MM-DD]` (daily detection activity totals),
   `/quality stream1|stream2|auto` (RTSP quality selection), and
   `/snapshot` — grabs every camera's latest live frame, replies with them as a
   photo album, and saves them to `data/server/collect/snapshots/`. Those saved
   frames are swept into the annotation pool by `uv run import-collect-birds`
   (the `snapshots` folder is imported by default).

## Practical Dataset Targets

Start with at least 100 to 200 labeled boxes per bird. Include day mode,
infrared mode, cage bars, floor, perches, partial occlusion, close-up shots,
far camera views, and frames with multiple birds.

Phone photos are useful for bootstrapping identities. Camera frames are more
important for validation because they match the deployment view.

## Runtime Notes

`uv run server` runs the server natively in the uv venv on the Linux/RTX 5060
host. `uv sync` already installs the cu128 torch build (pinned in
`pyproject.toml`), so there is no separate GPU-install step.

## External References

- Ultralytics YOLO dataset / label format: https://docs.ultralytics.com/datasets/detect/
- Ultralytics detection training: https://docs.ultralytics.com/tasks/detect/
- Tapo RTSP/ONVIF guidance: https://www.tapo.com/faq/34/
