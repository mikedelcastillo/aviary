#!/usr/bin/env bash
# Import collect/bird frames into models/annotation/raw/camera_frames (day/ir).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python models/annotation/scripts/import_collect_birds.py "$@"
