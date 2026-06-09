#!/usr/bin/env bash
uv sync
./scripts/install-gpu.sh
./scripts/install-clip.sh
uv run --no-sync generate-albums