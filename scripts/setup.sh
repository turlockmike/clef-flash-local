#!/usr/bin/env bash
# One-time setup: CUDA Python env at ~/.venvs/clef + Clef-flash weights (~18 GB in the HF cache).
# Windows: triton-windows provides Triton. Linux/WSL: swap triton-windows for triton.
set -e
V="$HOME/.venvs/clef"
uv venv "$V" --python 3.11 -q
PY="$V/Scripts/python.exe"; [ -x "$PY" ] || PY="$V/bin/python"
uv pip install -q --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cu130
TRITON=triton-windows; [ "$(uname -s)" = "Linux" ] && TRITON=triton
uv pip install -q --python "$PY" "transformers>=5.10.2" accelerate pillow huggingface_hub "$TRITON" flash-linear-attention torchao pytest
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Cloudflare/clef-flash'))"
"$PY" -c "import torch; assert torch.cuda.is_available(); print('ok', torch.cuda.get_device_name(0))"
