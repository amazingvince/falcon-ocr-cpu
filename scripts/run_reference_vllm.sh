#!/usr/bin/env bash
# Isolated, digest-pinned vLLM only; the image's pipeline service is not started.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
uuid=GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58
image=ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd
[[ $(nvidia-smi -i "$uuid" --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 4090' ]]
physical_index=$(nvidia-smi -i "$uuid" --query-gpu=index --format=csv,noheader,nounits)
[[ $physical_index =~ ^[0-9]+$ ]]
free_mib=$(nvidia-smi -i "$uuid" --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 12288 )) || { echo "Insufficient free RTX 4090 memory: $free_mib MiB" >&2; exit 1; }
output="$root/artifacts/reference/vllm-smoke-fp32"
args=(--model /models/Falcon-OCR --served-model-name falcon-ocr --host 127.0.0.1 --port 18080
  --max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.55 --trust-remote-code --dtype float32
  --attention-backend TRITON_ATTN
  --chat-template /app/falcon_ocr_chat_template.jinja --limit-mm-per-prompt '{"image":1}'
  --no-enable-chunked-prefill --enforce-eager)
if [[ ${1:-} == --full-pages ]]; then
  output="$root/artifacts/reference/vllm-fullpages-fp32-4096"
  [[ ! -e "$output" ]] || { echo "Full-page output already exists; preserve earlier attempts: $output" >&2; exit 1; }
  args+=(--hf-overrides '{"image_config":{"spatial_patch_size":16,"merge_size":1,"min_image_size":64,"max_image_size":1536,"min_pixels":3136,"max_pixels":10035200}}')
fi
if [[ ${1:-} == --probe ]]; then args=(--probe); fi
mkdir -p "$output"
bash "$root/scripts/podman_reference.sh" run --rm --pull=never \
  --name falcon-ocr-vllm-reference --network host --device /dev/dxg \
  --security-opt label=disable --shm-size 2g \
  --volume /usr/lib/wsl:/usr/lib/wsl:ro \
  --volume "$root:/workspace:ro" --volume "$root/artifacts/model:/models/Falcon-OCR:ro" \
  --volume "$output:/out:rw" \
  --env "CUDA_VISIBLE_DEVICES=$physical_index" --env "NVIDIA_VISIBLE_DEVICES=$uuid" \
  --env "FOCR_EXPECTED_GPU_UUID=$uuid" --env "FOCR_PHYSICAL_GPU_INDEX=$physical_index" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --env NVIDIA_TF32_OVERRIDE=0 --env TRITON_F32_DEFAULT=ieee \
  --env VLLM_FLOAT32_MATMUL_PRECISION=highest \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 --env HF_HUB_OFFLINE=1 \
  --env OMP_NUM_THREADS=8 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --entrypoint python "$image" /workspace/scripts/vllm_reference_entry.py "${args[@]}" \
  2>&1 | tee "$output/server.log"
