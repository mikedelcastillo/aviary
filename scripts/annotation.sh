#!/usr/bin/env bash
# Launch the custom bird annotation web tool on http://0.0.0.0:5000.
# Filesystem is the database: reads/writes annotations next to the raw images
# under data/annotation/raw, labels sourced from models/roster.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Absolute paths so the Next.js server resolves them regardless of cwd.
export AVIARY_DATA_ROOT="$REPO_ROOT/data/annotation/raw"
export AVIARY_ROSTER="$REPO_ROOT/models/roster.yaml"

cd annotation
if [ ! -d node_modules ]; then
  echo "Installing annotation tool dependencies (first run)..."
  npm install
fi

echo "Building annotation tool..."
npm run build

echo "Annotation tool -> http://0.0.0.0:5000"
exec npm run start
