# Aviary

Computer vision platform for monitoring a bird room with Tapo cameras.

The project is split into three working areas:

- `models/annotation/`: collect images and label birds with bounding boxes.
- `models/training/`: prepare datasets and train an Ultralytics YOLO detector.
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
| `uv run prepare-dataset` | normalize a CVAT YOLO export into a training dataset |
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

   - Put phone photos in `models/annotation/raw/phone_photos/`.
   - Pull likely bird photos from Immich with the standalone
     [immich-auto-albums](https://github.com/mikedelcastillo/immich-auto-albums) tool
     into `models/annotation/raw/immich_birds/`.
   - Extract Tapo day frames into `models/annotation/raw/camera_frames/day/`.
   - Extract Tapo infrared frames into `models/annotation/raw/camera_frames/ir/`.

3. Label images in CVAT using `models/annotation/roster.yaml` (the single label
   roster for both the live and archive models — see `models/README.md`).

4. Export the labeled CVAT task as YOLO and place it under
   `models/annotation/exports/<dataset_version>/`.

5. Prepare a YOLO dataset for the model you want (`--model` filters the roster to
   that model's classes and remaps the labels):

   ```bash
   uv run prepare-dataset \
     --source models/annotation/exports/v001 \
     --output models/training/datasets/v001 \
     --model live
   ```

6. Train:

   ```bash
   uv run train \
     --data models/training/datasets/v001/dataset.yaml \
     --epochs 100 \
     --export-to server/models/current/object_detector.pt
   ```

7. Set `TAPO_RSTP` in `.env` (cameras are defined in
   `server/lib/config.py`), then run the server:

   ```bash
   docker compose -f compose.dev.yml up --build
   ```

## Practical Dataset Targets

Start with at least 100 to 200 labeled boxes per bird. Include day mode,
infrared mode, cage bars, floor, perches, partial occlusion, close-up shots,
far camera views, and frames with multiple birds.

Phone photos are useful for bootstrapping identities. Camera frames are more
important for validation because they match the deployment view.

## Runtime Notes

The Dockerized server is the default deployment path for a Linux GPU machine.
On Apple Silicon, Docker usually will not expose MPS acceleration to PyTorch.
For that case, run the Python server directly in a local virtual environment
and set the model `device` to `mps` in `server/lib/config.py`.

## Immich Import

Bird-photo prefiltering from Immich now lives in its own repository:
[immich-auto-albums](https://github.com/mikedelcastillo/immich-auto-albums). Use its
`download-birds` command to pull each account's `Birds` album, then drop the images into
`models/annotation/raw/immich_birds/` for labeling.

## External References

- CVAT YOLO export format: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo/
- CVAT self-hosted installation: https://docs.cvat.ai/docs/administration/community/basics/installation/
- Ultralytics detection training: https://docs.ultralytics.com/tasks/detect/
- Tapo RTSP/ONVIF guidance: https://www.tapo.com/faq/34/
