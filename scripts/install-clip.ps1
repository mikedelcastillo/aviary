# Install open_clip (the optional CLIP scene model) into the uv venv WITHOUT touching torch.
# Run once, after `uv sync` and `scripts/install-gpu.ps1`. Then the app picks CLIP up automatically:
#
#     uv run --no-sync generate-albums      # IMMICH_CLIP defaults to "auto" -> on when open_clip is present
#
# Why a script instead of `uv sync --group clip`: open_clip_torch depends on torch/torchvision, so
# letting uv resolve it (any `uv sync`) would revert torch to the default build and undo
# install-gpu — the same reason torch itself is script-installed. So we install open_clip with
# --no-deps and add only its pure-Python deps, leaving the per-machine GPU torch (incl. the RX
# 7900XT ROCm build) untouched. CLIP then inherits the same fp16/device behavior as the detector.
# Turn CLIP back off without uninstalling by setting IMMICH_CLIP=0.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    # Pure-Python runtime deps (most already present transitively; idempotent, none pull torch).
    uv pip install regex ftfy tqdm huggingface-hub safetensors

    # open_clip itself + timm (a declared dep; ViT-B-32 doesn't use it but install it so any model
    # loads). --no-deps so neither drags in a torch/torchvision wheel over the GPU build.
    uv pip install --no-deps open_clip_torch timm

    Write-Host ""
    uv run --no-sync python -c "import torch, open_clip; print('open_clip', open_clip.__version__, '| torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
    Write-Host ""
    Write-Host "Done. CLIP is now on by default (IMMICH_CLIP=auto). Run:  uv run --no-sync generate-albums"
    Write-Host "Disable it anytime with:  `$env:IMMICH_CLIP=0 ; uv run --no-sync generate-albums"
}
finally {
    Pop-Location
}
