# Download preview-quality photos from each Immich "Birds" album into data/annotation/raw/phone.
# Equivalent to scripts/import_immich_albums.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv run --no-sync python training/scripts/import_immich_albums.py @args
}
finally {
    Pop-Location
}
