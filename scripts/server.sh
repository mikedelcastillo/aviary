#!/usr/bin/env bash
uv sync
./scripts/install-gpu.sh
uv run --no-sync server