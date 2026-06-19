# Build the live (real-time CCTV) bird detector end-to-end:
# prepare dataset from labeled raw data -> train -> export data/models/live-NNN.pt
# Equivalent to scripts/train_live.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    $Model = "live"

    uv sync
    & (Join-Path $PSScriptRoot "install-gpu.ps1")

    # Run the training scripts straight from source (not the force-included console
    # scripts `train`/`prepare-dataset`, which are *copied* into the venv at install
    # time and go stale after a `uv sync` until reinstalled). This matches
    # scripts/import_*.ps1. PYTHONPATH lets the source prepare_dataset.py import the
    # source aviary_training package.
    $trainingPath = Join-Path $RepoRoot "training"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$trainingPath;$env:PYTHONPATH"
    }
    else {
        $env:PYTHONPATH = $trainingPath
    }

    $Source = if ($env:AVIARY_LABEL_SOURCE) { $env:AVIARY_LABEL_SOURCE } else { "data/annotation/raw" }
    $Dataset = "data/training/datasets/$Model"

    if (Test-Path $Dataset) { Remove-Item -Recurse -Force $Dataset }
    uv run --no-sync python training/scripts/prepare_dataset.py `
        --source $Source `
        --output $Dataset `
        --model $Model

    # Export to data/models/<model>-NNN.pt, where NNN is the next zero-padded
    # sequence number after the highest existing one (so each run is preserved and
    # never clobbers the model the server may currently be loading).
    $ModelsDir = "data/models"
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    $last = 0
    Get-ChildItem -Path $ModelsDir -Filter "$Model-*.pt" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.BaseName -match "^$Model-(\d{3})$") {
            $n = [int]$Matches[1]
            if ($n -gt $last) { $last = $n }
        }
    }
    $Export = Join-Path $ModelsDir ("{0}-{1:D3}.pt" -f $Model, ($last + 1))

    uv run --no-sync python training/scripts/train.py `
        --data "$Dataset/dataset.yaml" `
        --name $Model `
        --export-to $Export `
        @args

    Write-Host "Exported $Model model to $Export"
}
finally {
    Pop-Location
}
