# Propose bird bounding boxes for annotation images and write them to the
# <image>.suggest.json sidecar (approve/reject as yellow boxes in Box mode).
# Defaults to generic yolo11n.pt (COCO 'bird' class). Equivalent to
# scripts/suggest_boxes.sh on Windows. Extra flags pass through to the script.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv run --no-sync python training/scripts/suggest_boxes.py @args
}
finally {
    Pop-Location
}
