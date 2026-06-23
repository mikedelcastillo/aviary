# Training

This folder prepares YOLO datasets and trains the bird detectors. The label
strategy and class scheme live in [`STRATEGY.md`](STRATEGY.md); the single label
roster is [`roster.yaml`](roster.yaml).

Two models are built from that one roster:

- `live` — 6 living birds + 3 IR species + `unknown_bird` (the real-time CCTV detector).
- `archive` — all individuals (living + deceased), no species (the photo-library catalog).

Versioned weights land in `data/models/`: `live-NNN.pt` and `archive-NNN.pt`
(gitignored), where `NNN` is the next zero-padded sequence number after the
highest existing one — so each run is preserved and never clobbers the model the
server may currently be loading.

## One-command build (recommended)

From the repo root, build a model end-to-end — prepare its dataset from the
labeled raw images, train, and export the weights:

```bash
uv run train-live      # -> data/models/live-NNN.pt
uv run train-archive   # -> data/models/archive-NNN.pt
```

Each command prepares `data/training/datasets/<model>/` from
`data/annotation/raw`, then trains and exports to the next sequence number.
(`uv sync` already installed the cu128 torch build for the RTX 5060.) Extra flags
pass through to `train`, e.g.:

```bash
uv run train-live --epochs 200 --imgsz 960 --device cuda:0 --model yolo11s.pt
```

Override the labeled-image source with `AVIARY_LABEL_SOURCE` if your layout differs.

## Run the steps by hand

The scripts above are thin wrappers over the same console commands.

### Prepare a dataset

`--model` filters [`roster.yaml`](roster.yaml) to that model's classes and remaps
the label indices to a contiguous range (dropping classes the model does not use):

```bash
uv run prepare-dataset \
  --source data/annotation/raw \
  --output data/training/datasets/live \
  --model live      # or: --model archive
```

The `/annotation` tool writes YOLO `.txt` labels directly next to each raw image,
so `data/annotation/raw` *is* the export — there is no separate export step. This
creates:

```text
data/training/datasets/live/
  dataset.yaml
  images/{train,val,test}
  labels/{train,val,test}
  manifest.csv
```

### Train

```bash
uv run train \
  --data data/training/datasets/live/dataset.yaml \
  --epochs 100 \
  --imgsz 960 \
  --model yolo11n.pt \
  --name live \
  --export-to training/models/live.pt
```

Use `--device cuda:0` to pin the RTX 5060, or omit `--device` to let Ultralytics
decide. `--name live`/`archive` keeps each model's runs separate under
`data/training/runs/`.

### Evaluate

```bash
uv run evaluate \
  --model training/models/live.pt \
  --data data/training/datasets/live/dataset.yaml
```

Track day and infrared results separately. If IR performance is weak, add more
IR frames before tuning model architecture.
