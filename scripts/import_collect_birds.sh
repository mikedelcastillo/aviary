#!/usr/bin/env bash
# Import data/server/collect/bird frames into data/annotation/raw/tapo (day/ir).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python training/scripts/import_collect_birds.py "$@"
