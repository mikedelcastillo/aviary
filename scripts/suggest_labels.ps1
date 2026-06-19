# Suggest a roster identity for each box (human-drawn or proposed) on annotation
# images, writing label suggestions into the <image>.suggest.json sidecar
# (pre-highlighted pill in Label mode). Uses the latest data/models/live-NNN.pt.
# Equivalent to scripts/suggest_labels.sh. Extra flags pass through to the script.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv run --no-sync python training/scripts/suggest_labels.py @args
}
finally {
    Pop-Location
}
