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
ENTRYPOINT=scripts/export_reference.py
if [[ "${1:-}" == "--official" ]]; then
    ENTRYPOINT=scripts/export_official_reference.py
    shift
elif [[ "${1:-}" == "--official-fullpages" ]]; then
    ENTRYPOINT=scripts/run_official_fullpages.py
    shift
elif [[ "${1:-}" == "--operators" ]]; then
    ENTRYPOINT=scripts/export_operator_reference.py
    shift
elif [[ "${1:-}" == "--attention-operators" ]]; then
    ENTRYPOINT=scripts/export_attention_reference.py
    shift
elif [[ "${1:-}" == "--rope-operators" ]]; then
    ENTRYPOINT=scripts/export_rope_reference.py
    shift
elif [[ "${1:-}" == "--corpus" ]]; then
    ENTRYPOINT=scripts/run_corpus_reference.py
    shift
elif [[ "${1:-}" == "--linear-operators" ]]; then
    ENTRYPOINT=scripts/export_linear_reference.py
    shift
elif [[ "${1:-}" == "--bf16-operators" ]]; then
    ENTRYPOINT=scripts/export_bf16_operators.py
    shift
elif [[ "${1:-}" == "--bf16-blockwise" ]]; then
    ENTRYPOINT=scripts/verify_bf16_blockwise.py
    shift
elif [[ "${1:-}" == "--bf16-freeze-local" ]]; then
    ENTRYPOINT=scripts/freeze_bf16_local_contract.py
    shift
elif [[ "${1:-}" == "--layer-operators" ]]; then
    ENTRYPOINT=scripts/export_layer_operators.py
    shift
elif [[ "${1:-}" == "--bf16-attention-tiles" ]]; then
    ENTRYPOINT=scripts/export_bf16_attention_tiles.py
    shift
elif [[ "${1:-}" == "--rms-rstd" ]]; then
    ENTRYPOINT=scripts/export_rms_rstd_reference.py
    shift
elif [[ "${1:-}" == "--rsqrt-replay" ]]; then
    ENTRYPOINT=scripts/export_rsqrt_replay_reference.py
    shift
elif [[ "${1:-}" == "--rsqrt-table" ]]; then
    ENTRYPOINT=experiments/rms_norm/export_rsqrt_table.py
    shift
elif [[ "${1:-}" == "--linear-profiler" ]]; then
    ENTRYPOINT=scripts/profile_linear_reference.py
    shift
elif [[ "${1:-}" == "--linear-logger" ]]; then
    ENTRYPOINT=scripts/profile_linear_logging_reference.py
    shift
elif [[ "${1:-}" == "--owned-linear-workspace" ]]; then
    ENTRYPOINT=experiments/linear/export_owned_workspace.py
    shift
elif [[ "${1:-}" == "--bf16-fused-substages" ]]; then
    ENTRYPOINT=scripts/export_bf16_fused_substages.py
    shift
elif [[ "${1:-}" == "--bf16-exp2-replay" ]]; then
    ENTRYPOINT=scripts/export_bf16_exp2_replay.py
    shift
elif [[ "${1:-}" == "--bf16-ptx-roundtrip" ]]; then
    ENTRYPOINT=scripts/export_bf16_ptx_roundtrip.py
    shift
elif [[ "${1:-}" == "--bf16-ptx-observer" ]]; then
    ENTRYPOINT=scripts/export_bf16_ptx_observer.py
    shift
fi
exec "$ENV_ROOT/.venv/bin/python" "$ENTRYPOINT" "$@"
