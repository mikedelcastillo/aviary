# Install the correct PyTorch GPU build for THIS machine into the uv venv.
# Run once, after `uv sync`. Then run the app with:
#
#     uv run --no-sync train
#
# The `--no-sync` matters: a plain `uv run`/`uv sync` reverts torch to the default build (uv
# doesn't know about the GPU wheel installed here). To drop the flag, set UV_NO_SYNC=1 once
# (e.g. [Environment]::SetEnvironmentVariable("UV_NO_SYNC","1","User")) and reopen the shell.
#
# RX 7900XT (AMD) uses native-Windows ROCm wheels from repo.radeon.com. Prerequisites, one time:
#   * Python 3.12 (the ROCm wheels are cp312-only) — use a 3.12 venv: uv venv --python 3.12
#   * AMD Software / Adrenalin graphics driver 26.2.2 or newer
# Source: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    $gpus = (Get-CimInstance Win32_VideoController).Name -join "; "
    Write-Host "Detected display adapters: $gpus"

    if ($gpus -match "Radeon|AMD") {
        $Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $Py)) { throw "Missing .venv. Run 'uv sync' first." }
        $ver = (& $Py -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
        if ($ver -ne "3.12") {
            throw "ROCm Windows wheels need Python 3.12, but the venv is $ver. Recreate it: uv venv --python 3.12 ; uv sync"
        }
        $Base = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
        Write-Host "Installing ROCm 7.2.1 SDK runtime + torch (RDNA3, e.g. RX 7900XT)..."
        & $Py -m pip install --no-cache-dir `
            "$Base/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
            "$Base/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
            "$Base/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
            "$Base/rocm-7.2.1.tar.gz"
        & $Py -m pip install --no-cache-dir `
            "$Base/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
            "$Base/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
    }
    elseif ($gpus -match "NVIDIA|GeForce") {
        if ($gpus -match "GTX 10\d\d") { $url = "https://download.pytorch.org/whl/cu118" }
        else { $url = "https://download.pytorch.org/whl/cu128" }
        Write-Host "Installing torch from $url"
        uv pip install --reinstall torch torchvision --index-url $url
    }
    else {
        Write-Host "No supported GPU detected; keeping the default CPU torch from 'uv sync'."
        exit 0
    }

    Write-Host ""
    uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
    Write-Host ""
    Write-Host "Done. Run the app with:  uv run --no-sync train"
}
finally {
    Pop-Location
}
