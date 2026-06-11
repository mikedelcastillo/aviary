#!/usr/bin/env bash
# Download preview-quality photos from each Immich "Birds" album into data/annotation/raw/phone.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python training/scripts/import_immich_albums.py "$@"
