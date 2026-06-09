#!/usr/bin/env bash
# Install open_clip (the optional CLIP scene model) into the uv venv WITHOUT touching torch.
# Run once, after `uv sync` and `scripts/install-gpu.sh`. Then the app picks CLIP up automatically:
#
#     uv run --no-sync generate-albums      # IMMICH_CLIP defaults to "auto" -> on when open_clip is present
#
# Why a script instead of `uv sync --group clip`: open_clip_torch depends on torch/torchvision, so
# letting uv resolve it (any `uv sync`) would revert torch to the default build and undo
# install-gpu — the same reason torch itself is script-installed. So we install open_clip with
# --no-deps and add only its pure-Python deps, leaving the per-machine GPU torch untouched.
#
# CLIP rides whatever torch build install-gpu put in the venv (cu118/cu128/ROCm), so it inherits the
# same fp16/device behavior as the YOLO detector. To turn CLIP back off without uninstalling, set
# IMMICH_CLIP=0.
set -euo pipefail

cd "$(dirname "$0")/.."

# Pure-Python runtime deps open_clip needs (most already present transitively; safe + idempotent,
# none of these pull torch).
uv pip install regex ftfy tqdm huggingface-hub safetensors

# open_clip itself + timm (a declared dep; ViT-B-32 doesn't use it but install it so any model
# loads). --no-deps so neither drags in a torch/torchvision wheel over the GPU build.
uv pip install --no-deps open_clip_torch timm

echo
uv run --no-sync python - <<'PY'
import torch, open_clip
print("open_clip", open_clip.__version__, "| torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
PY
echo
echo "Done. CLIP is now on by default (IMMICH_CLIP=auto). Run:  uv run --no-sync generate-albums"
echo "Disable it anytime with:  IMMICH_CLIP=0 uv run --no-sync generate-albums"
