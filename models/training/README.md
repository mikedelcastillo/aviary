# Training

This folder prepares YOLO-format label exports and trains the bird detector.

## Setup

From the repo root (see the top-level README for installing uv):

```bash
uv sync
```

## Prepare a Dataset

Place a YOLO-format label export under `data/annotation/exports/v001`, then normalize
it for the model you are building. `--model` filters `models/roster.yaml`
to that model's classes and remaps the exported label indices to a contiguous
range (dropping classes the model does not use):

```bash
uv run prepare-dataset \
  --source data/annotation/exports/v001 \
  --output data/training/datasets/v001-live \
  --model live      # or: --model archive
```

- `live` — 6 living birds + 3 IR species + `unknown_bird` (the CCTV detector).
- `archive` — all individuals (living + deceased), no species (photo catalog).

You label everything once against the full roster; this step produces each
model's dataset. See [`../README.md`](../README.md) for the two-model rationale.

This creates:

```text
data/training/datasets/v001/
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
  --data data/training/datasets/v001/dataset.yaml \
  --epochs 100 \
  --imgsz 960 \
  --model yolo11n.pt \
  --export-to data/server/models/current/bird_detector.pt
```

Use `--device cuda:0` on a Linux NVIDIA machine or `--device mps` on Apple
Silicon. Omit `--device` to let Ultralytics decide.

## Evaluate

```bash
uv run evaluate \
  --model data/server/models/current/bird_detector.pt \
  --data data/training/datasets/v001/dataset.yaml
```

Track day and infrared results separately. If IR performance is weak, add more
IR frames before tuning model architecture.
