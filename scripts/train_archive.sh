#!/usr/bin/env bash
# Build the archive (photo-library catalog) bird detector end-to-end:
# prepare dataset from labeled raw data -> train -> export data/models/archive-NNN.pt
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=archive

uv sync
./scripts/install-gpu.sh

# Run the training scripts straight from source (not the force-included console
# scripts `train`/`prepare-dataset`, which are *copied* into the venv at install
# time and go stale after a `uv sync` until reinstalled). This matches
# scripts/import_*.sh. PYTHONPATH lets the source prepare_dataset.py import the
# source aviary_training package.
export PYTHONPATH="$PWD/training${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="${AVIARY_LABEL_SOURCE:-data/annotation/raw}"
DATASET="data/training/datasets/$MODEL"

rm -rf "$DATASET"
uv run --no-sync python training/scripts/prepare_dataset.py \
  --source "$SOURCE" \
  --output "$DATASET" \
  --model "$MODEL"

# Export to data/models/<model>-NNN.pt, where NNN is the next zero-padded
# sequence number after the highest existing one (so each run is preserved and
# never clobbers the model the server may currently be loading).
MODELS_DIR="data/models"
mkdir -p "$MODELS_DIR"
last=0
for f in "$MODELS_DIR/$MODEL"-[0-9][0-9][0-9].pt; do
  [ -e "$f" ] || continue
  n=${f##*-}; n=${n%.pt}
  n=$((10#$n))
  ((n > last)) && last=$n
done
EXPORT="$MODELS_DIR/$MODEL-$(printf '%03d' "$((last + 1))").pt"

uv run --no-sync python training/scripts/train.py \
  --data "$DATASET/dataset.yaml" \
  --name "$MODEL" \
  --export-to "$EXPORT" \
  "$@"

echo "Exported $MODEL model to $EXPORT"
