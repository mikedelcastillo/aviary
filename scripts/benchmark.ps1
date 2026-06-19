# Benchmark every model in data/models against the labeled annotation boxes and
# write data/models/benchmark.json (per-label / per-model scores, grouped into
# the live and archive series). live-* models are scored on tapo/day + tapo/ir;
# archive-* models on phone. See training/scripts/benchmark.py.
# Equivalent to scripts/benchmark.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    uv sync
    & (Join-Path $PSScriptRoot "install-gpu.ps1")

    # Run the script straight from source (not the force-included console script,
    # which is copied into the venv at install time and goes stale after a uv sync).
    # PYTHONPATH matches the other wrappers for consistency.
    $trainingPath = Join-Path $RepoRoot "training"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$trainingPath;$env:PYTHONPATH"
    }
    else {
        $env:PYTHONPATH = $trainingPath
    }

    uv run --no-sync python training/scripts/benchmark.py @args
}
finally {
    Pop-Location
}
