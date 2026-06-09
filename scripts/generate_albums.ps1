# Sync deps, install GPU torch + CLIP, then run generate-albums.
# Equivalent to scripts/generate_albums.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv sync
    & (Join-Path $PSScriptRoot "install-gpu.ps1")
    & (Join-Path $PSScriptRoot "install-clip.ps1")
    uv run --no-sync generate-albums
}
finally {
    Pop-Location
}
