#!/usr/bin/env bash
# Install the correct PyTorch GPU build for THIS machine into the uv venv.
# Run once, after `uv sync`. Then run the app with:
#
#     uv run --no-sync train
#
# The `--no-sync` matters: a plain `uv run`/`uv sync` reverts torch to the default CPU build
# (uv doesn't know about the GPU wheel installed here). To drop the flag, set UV_NO_SYNC=1 once
# in your shell profile and then `uv run train` works as-is.
#
# uv can't pick the build automatically — the GTX 1060 and RTX 5060 both look like
# "linux/x86_64" to uv but need incompatible wheels (cu118/Pascal vs cu128/Blackwell).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU detected. The default CPU torch from 'uv sync' will be used."
    echo "(AMD on Windows: use scripts/install-gpu.ps1 instead.)"
    exit 0
fi

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
major="${cap%%.*}"
if [ "${major:-0}" -le 6 ]; then
    URL="https://download.pytorch.org/whl/cu118"
    LABEL="cu118 (Pascal/Maxwell, e.g. GTX 1060)"
else
    URL="https://download.pytorch.org/whl/cu128"
    LABEL="cu128 (Turing..Blackwell, e.g. RTX 5060)"
fi

echo "Detected compute capability ${cap:-unknown} -> installing torch ${LABEL}"
uv pip install --reinstall torch torchvision --index-url "${URL}"

echo
uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo
echo "Done. Run the app with:  uv run --no-sync train"
echo "(or 'export UV_NO_SYNC=1' once, then just 'uv run train')"
