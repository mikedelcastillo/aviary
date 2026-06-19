# Sync deps, install the per-machine GPU torch build, then run the server.
# Equivalent to scripts/server.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv sync
    & (Join-Path $PSScriptRoot "install-gpu.ps1")
    uv run --no-sync server
}
finally {
    Pop-Location
}
