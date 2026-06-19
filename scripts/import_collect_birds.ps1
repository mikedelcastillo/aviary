# Import data/server/collect/{bird + every roster name} frames into
# data/annotation/raw/tapo (day/ir). Folder list comes from training/roster.yaml.
# Equivalent to scripts/import_collect_birds.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv run --no-sync python training/scripts/import_collect_birds.py @args
}
finally {
    Pop-Location
}
