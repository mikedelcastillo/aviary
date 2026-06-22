# Propose boxes AND roster labels for annotation images in one pass per image,
# writing combined, de-duplicated proposals to the <image>.suggest.json sidecar
# (approve/reject as yellow boxes/pills in the annotation tool). Uses a generic
# yolo11n.pt (COCO 'bird') plus the latest data/models/live-NNN.pt. Equivalent to
# scripts/suggest.sh on Windows. Extra flags pass through to the script.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv run --no-sync python training/scripts/suggest.py @args
}
finally {
    Pop-Location
}
