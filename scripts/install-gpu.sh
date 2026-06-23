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
# uv can't pick the build automatically — it sees only "linux/x86_64" and not the actual GPU,
# so it can't choose the right CUDA wheel (e.g. cu128/Blackwell for the RTX 5060).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU detected. The default CPU torch from 'uv sync' will be used."
    echo "(AMD on Windows: use scripts/install-gpu.ps1 instead.)"
    exit 0
fi

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
URL="https://download.pytorch.org/whl/cu128"
LABEL="cu128 (Turing..Blackwell, e.g. RTX 5060)"

echo "Detected compute capability ${cap:-unknown} -> installing torch ${LABEL}"
uv pip install --reinstall torch torchvision --index-url "${URL}"

echo
uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo
echo "Done. Run the app with:  uv run --no-sync train"
echo "(or 'export UV_NO_SYNC=1' once, then just 'uv run train')"
