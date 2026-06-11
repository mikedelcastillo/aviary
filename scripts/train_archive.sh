#!/usr/bin/env bash
# Build the archive (photo-library catalog) bird detector end-to-end:
# prepare dataset from labeled raw data -> train -> export training/models/archive.pt
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
./scripts/install-gpu.sh

# Run the training scripts straight from source (not the force-included console
# scripts `train`/`prepare-dataset`, which are *copied* into the venv at install
# time and go stale after a `uv sync` until reinstalled). This matches
# scripts/import_*.sh. PYTHONPATH lets the source prepare_dataset.py import the
# source aviary_training package.
export PYTHONPATH="$PWD/training${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="${AVIARY_LABEL_SOURCE:-data/annotation/raw}"
DATASET="data/training/datasets/archive"

rm -rf "$DATASET"
uv run --no-sync python training/scripts/prepare_dataset.py \
  --source "$SOURCE" \
  --output "$DATASET" \
  --model archive

mkdir -p training/models
uv run --no-sync python training/scripts/train.py \
  --data "$DATASET/dataset.yaml" \
  --name archive \
  --export-to training/models/archive.pt \
  "$@"
