#!/usr/bin/env bash
# Propose boxes AND roster labels for annotation images in one pass per image,
# writing combined, de-duplicated proposals to the <image>.suggest.json sidecar
# (approve/reject as yellow boxes/pills in the annotation tool). Uses a generic
# yolo11n.pt (COCO 'bird') plus the latest data/models/live-NNN.pt. Extra flags
# pass through to the script.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python training/scripts/suggest.py "$@"
