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
#   * Python 3.12 (the ROCm wheels are cp312-only) -- use a 3.12 venv: uv venv --python 3.12
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
        # Use `uv pip` (not `$Py -m pip`): uv-created venvs ship no pip, so `python -m pip`
        # fails with "No module named pip". `--reinstall` forces the ROCm wheels over the
        # default CPU torch from `uv sync`.
        uv pip install --no-cache --reinstall `
            "$Base/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
            "$Base/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
            "$Base/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
            "$Base/rocm-7.2.1.tar.gz"
        if ($LASTEXITCODE -ne 0) { throw "ROCm SDK install failed (uv exit $LASTEXITCODE)." }
        # --no-deps: the torch ROCm wheel declares a dependency on `rocm[libraries]==7.2.1`, an
        # extra whose provider (rocm-sdk-libraries-custom) is served from repo.radeon.com, not
        # PyPI, so uv cannot resolve it and aborts. The full ROCm runtime is already installed
        # above, so skip dependency resolution for the torch/torchvision wheels themselves.
        uv pip install --no-cache --reinstall --no-deps `
            "$Base/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
            "$Base/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
        if ($LASTEXITCODE -ne 0) { throw "ROCm torch install failed (uv exit $LASTEXITCODE)." }
    }
    elseif ($gpus -match "NVIDIA|GeForce") {
        $url = "https://download.pytorch.org/whl/cu128"
        Write-Host "Installing torch from $url"
        # cu128 covers the current NVIDIA card (RTX 5060, Blackwell / sm_120). No version pin: the
        # CUDA index serves its own newest torch and uv already picks it. The smoke import below
        # fails loudly on an incompatible torch/torchvision/ultralytics combo.
        uv pip install --reinstall torch torchvision --index-url $url
        if ($LASTEXITCODE -ne 0) { throw "torch install failed (uv exit $LASTEXITCODE)." }
    }
    else {
        Write-Host "No supported GPU detected; keeping the default CPU torch from 'uv sync'."
        # `return` (not `exit`): train_live.ps1 / server.ps1 call this in-process via
        # `& install-gpu.ps1`, so `exit` would abort the caller too.
        return
    }

    Write-Host ""
    # Fail loudly if the swapped torch/torchvision/ultralytics combo is import- or ABI-incompatible.
    # Retry: right after a fresh ROCm install the first `import torch` can transiently fail with
    # "No module named 'rocm_sdk'" before the just-installed runtime is visible to a new subprocess.
    # A second attempt succeeds, so retry a few times before giving up (prevents callers like
    # train_live.ps1 / server.ps1 from aborting on a false-negative).
    $smoke = "import torch, torchvision, ultralytics; from torchvision.ops import nms; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda available:', torch.cuda.is_available())"
    $ok = $false
    for ($i = 1; $i -le 3; $i++) {
        uv run --no-sync python -c $smoke
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Write-Host "Smoke test attempt $i failed (exit $LASTEXITCODE); retrying in 3s..."
        Start-Sleep -Seconds 3
    }
    if (-not $ok) { throw "torch/torchvision/ultralytics smoke test failed after 3 attempts." }
    Write-Host ""
    Write-Host "Done. Run the app with:  uv run --no-sync train"
}
finally {
    Pop-Location
}
