#!/usr/bin/env bash
# Independent GPU-only references. Dense BF16 is diagnostic, never a parity gate.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
reference_python=${FOCR_REFERENCE_ENV:-/home/amazi/falcon-ocr-rust-reference}/.venv/bin/python
bash scripts/run_reference.sh --precision bf16 --image artifacts/reference/smoke-fp32/input.png \
  --output artifacts/reference/smoke-bf16 --max-new-tokens 24 --max-dimension 256
bash scripts/run_reference.sh --precision bf16 --attention dense \
  --image artifacts/reference/smoke-fp32/input.png --teacher-tokens artifacts/reference/smoke-bf16/teacher-tokens.json \
  --output artifacts/reference/smoke-dense-bf16 --max-new-tokens 24 --max-dimension 256
bash scripts/run_reference.sh --attention-operators --precision bf16 \
  --source artifacts/reference/smoke-bf16/trace.safetensors \
  --output artifacts/reference/attention-operators-bf16.safetensors
exec "$reference_python" scripts/compare_traces.py \
  artifacts/reference/smoke-bf16/trace.safetensors artifacts/reference/smoke-dense-bf16/trace.safetensors \
  --diagnostic --output reference/gpu-dense-smoke-bf16-diagnostic.json
