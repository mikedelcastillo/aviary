#!/usr/bin/env bash
# Propose bird bounding boxes for annotation images and write them to the
# <image>.suggest.json sidecar (approve/reject as yellow boxes in Box mode).
# Defaults to generic yolo11n.pt (COCO 'bird' class). Extra flags pass through.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python training/scripts/suggest_boxes.py "$@"
