#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_ROOT=${FOCR_REFERENCE_ENV:-/home/amazi/falcon-ocr-rust-reference}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58
export HF_HOME="$ENV_ROOT/cache/huggingface"
export TRITON_CACHE_DIR="$ENV_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ENV_ROOT/cache/torchinductor"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
"$ENV_ROOT/.venv/bin/python" scripts/reference_preflight.py
exec "$ENV_ROOT/.venv/bin/python" research/bf16-graph/scripts/export_bf16_ptx_observer_v2.py "$@"
