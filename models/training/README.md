# Training

This folder prepares CVAT YOLO exports and trains the bird detector.

## Setup

From the repo root (see the top-level README for installing uv):

```bash
uv sync
```

## Prepare a Dataset

Place a CVAT YOLO export under `models/annotation/exports/v001`, then normalize it:

```bash
uv run prepare-dataset \
  --source models/annotation/exports/v001 \
  --output models/training/datasets/v001
```

This creates:

```text
models/training/datasets/v001/
  dataset.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  manifest.csv
```

## Train

```bash
uv run train \
  --data models/training/datasets/v001/dataset.yaml \
  --epochs 100 \
  --imgsz 960 \
  --model yolo11n.pt \
  --export-to server/models/current/bird_detector.pt
```

Use `--device cuda:0` on a Linux NVIDIA machine or `--device mps` on Apple
Silicon. Omit `--device` to let Ultralytics decide.

## Evaluate

```bash
uv run evaluate \
  --model server/models/current/bird_detector.pt \
  --data models/training/datasets/v001/dataset.yaml
```

Track day and infrared results separately. If IR performance is weak, add more
IR frames before tuning model architecture.
