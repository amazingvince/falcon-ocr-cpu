#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_ROOT=${FOCR_REFERENCE_ENV:-/home/amazi/falcon-ocr-rust-reference}
UV=${UV:-/home/amazi/.local/bin/uv}
if [[ ! -f "$ENV_ROOT/.venv/pyvenv.cfg" ]]; then
    "$UV" venv --python 3.12.13 "$ENV_ROOT/.venv"
fi
if [[ -f "$ROOT/requirements/reference-lock.txt" ]]; then
    "$UV" pip sync --python "$ENV_ROOT/.venv/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 "$ROOT/requirements/reference-lock.txt"
else
    "$UV" pip install --python "$ENV_ROOT/.venv/bin/python" --index-strategy unsafe-best-match -r "$ROOT/requirements/reference.txt"
    "$UV" pip freeze --python "$ENV_ROOT/.venv/bin/python" > "$ROOT/requirements/reference-lock.txt"
fi
mkdir -p "$ENV_ROOT/cache/huggingface" "$ENV_ROOT/cache/triton" "$ENV_ROOT/cache/torchinductor"
echo "Reference Python: $ENV_ROOT/.venv/bin/python"
