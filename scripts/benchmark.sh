#!/usr/bin/env bash
# Benchmark every model in data/models against the labeled annotation boxes and
# write data/models/benchmark.json (per-label / per-model scores, grouped into
# the live and archive series). live-* models are scored on tapo/day + tapo/ir;
# archive-* models on phone. See training/scripts/benchmark.py.
# Equivalent to scripts/benchmark.ps1 on Windows.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
./scripts/install-gpu.sh

# Run the script straight from source (not the force-included console script,
# which is copied into the venv at install time and goes stale after a uv sync).
# PYTHONPATH matches the other wrappers for consistency.
export PYTHONPATH="$PWD/training${PYTHONPATH:+:$PYTHONPATH}"

# Default to the held-out TEST split so the homepage isn't scoring training
# images (pass --split all to score the whole labeled tree).
uv run --no-sync python training/scripts/benchmark.py --split test "$@"
