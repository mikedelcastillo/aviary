# Aviary

Computer vision platform for monitoring a bird room with Tapo cameras.

The project is split into three working areas:

- `models/annotation/`: collect images and label birds with bounding boxes.
- `models/immich/`: scan Immich libraries for likely bird photos and download them.
- `models/training/`: prepare datasets and train an Ultralytics YOLO detector.
- `server/`: consume camera streams, run inference, and send Telegram alerts.

The first production goal is to identify each of the six birds. The model uses
one detection class per bird plus `unknown_bird` for visible birds that are not
confidently identifiable.

## Quick Start

1. Copy the environment template and fill in secrets:

   ```bash
   cp .env.example .env
   ```

2. Collect seed images:

   - Put phone photos in `models/annotation/raw/phone_photos/`.
   - Pull likely bird photos from Immich with `models/immich/generate_albums.py` and
     `models/immich/download_immich_birds.py`.
   - Extract Tapo day frames into `models/annotation/raw/camera_frames/day/`.
   - Extract Tapo infrared frames into `models/annotation/raw/camera_frames/ir/`.

3. Label images in CVAT using `models/annotation/labeling_schema.yaml`.

4. Export the labeled CVAT task as YOLO and place it under
   `models/annotation/exports/<dataset_version>/`.

5. Prepare a YOLO dataset:

   ```bash
   python models/training/scripts/prepare_dataset.py \
     --source models/annotation/exports/v001 \
     --output models/training/datasets/v001
   ```

6. Train:

   ```bash
   python models/training/scripts/train.py \
     --data models/training/datasets/v001/dataset.yaml \
     --epochs 100 \
     --export-to server/models/current/bird_detector.pt
   ```

7. Configure cameras in `server/config/cameras.yaml`, then run the server:

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
and set `model.device: mps` in `server/config/cameras.yaml`.

## Immich Import

Create one Immich API key per account, copy
`models/immich/config/accounts.example.yaml` to `models/immich/config/accounts.yaml`, and
fill `.env` with `IMMICH_BASE_URL` plus the account API keys.

Start with:

```bash
python models/immich/generate_albums.py --dry-run --limit 25
```

Then generate real `Birds` albums and download them:

```bash
python models/immich/generate_albums.py
python models/immich/download_immich_birds.py
```

## External References

- CVAT YOLO export format: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo/
- CVAT self-hosted installation: https://docs.cvat.ai/docs/administration/community/basics/installation/
- Ultralytics detection training: https://docs.ultralytics.com/tasks/detect/
- Tapo RTSP/ONVIF guidance: https://www.tapo.com/faq/34/
