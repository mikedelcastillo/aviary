#!/usr/bin/env bash
# Import data/server/collect/{bird + every roster name} frames into
# data/annotation/raw/tapo (day/ir). Folder list comes from training/roster.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python training/scripts/import_collect_birds.py "$@"
