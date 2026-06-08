# Training

This folder prepares CVAT YOLO exports and trains the bird detector.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r models/training/requirements.txt
```

## Prepare a Dataset

Place a CVAT YOLO export under `models/annotation/exports/v001`, then normalize it:

```bash
python models/training/scripts/prepare_dataset.py \
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
python models/training/scripts/train.py \
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
python models/training/scripts/evaluate.py \
  --model server/models/current/bird_detector.pt \
  --data models/training/datasets/v001/dataset.yaml
```

Track day and infrared results separately. If IR performance is weak, add more
IR frames before tuning model architecture.
