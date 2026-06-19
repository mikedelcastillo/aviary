#!/usr/bin/env bash
# Suggest a roster identity for each box (human-drawn or proposed) on annotation
# images, writing label suggestions into the <image>.suggest.json sidecar
# (pre-highlighted pill in Label mode). Uses the latest data/models/live-NNN.pt.
# Extra flags pass through to the script.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python training/scripts/suggest_labels.py "$@"
