$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Missing .venv Python at $PythonExe. Create the virtual environment before running this script."
}

Push-Location $RepoRoot
try {
    & $PythonExe "models\immich\generate_albums.py"
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
