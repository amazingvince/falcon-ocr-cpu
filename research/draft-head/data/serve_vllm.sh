#!/usr/bin/env bash
# Draft-head transcript server: the production BF16 vLLM image and flags,
# with at most 128 concurrent sequences (bounded host memory), on the RTX 4090
# by default; GPU_UUID, GPU_NAME, PORT and NAME select another GPU/instance.
#   bash research/draft-head/data/serve_vllm.sh <output dir>
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
output=${1:?output directory}
uuid=${GPU_UUID:-GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58}
gpu_name=${GPU_NAME:-NVIDIA GeForce RTX 4090}
port=${PORT:-18081}
name=${NAME:-falcon-ocr-vllm-draftgen}
image=ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd
[[ $(nvidia-smi -i "$uuid" --query-gpu=name --format=csv,noheader) == "$gpu_name" ]]
physical_index=$(nvidia-smi -i "$uuid" --query-gpu=index --format=csv,noheader,nounits)
mkdir -p "$output"
args=(--model /models/Falcon-OCR --served-model-name falcon-ocr --host 127.0.0.1 --port "$port"
  --max-model-len 16384 --max-num-seqs 128 --gpu-memory-utilization 0.85
  --trust-remote-code --dtype bfloat16
  --chat-template /app/falcon_ocr_chat_template.jinja --limit-mm-per-prompt '{"image":1}'
  --no-enable-chunked-prefill --max-logprobs 32
  --hf-overrides '{"image_config":{"spatial_patch_size":16,"merge_size":1,"min_image_size":64,"max_image_size":1536,"min_pixels":3136,"max_pixels":10035200}}')
bash "$root/research/vllm-serving/scripts/podman_reference.sh" run --rm --pull=never \
  --name "$name" --network host --device /dev/dxg \
  --security-opt label=disable --shm-size 2g \
  --volume /usr/lib/wsl:/usr/lib/wsl:ro \
  --volume "$root:/workspace:ro" --volume "$root/artifacts/model:/models/Falcon-OCR:ro" \
  --volume "$output:/out:rw" \
  --env "CUDA_VISIBLE_DEVICES=$physical_index" --env "NVIDIA_VISIBLE_DEVICES=$uuid" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --env VLLM_ATTENTION_BACKEND=TRITON_ATTN --env VLLM_FLOAT32_MATMUL_PRECISION=high \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 --env HF_HUB_OFFLINE=1 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint python "$image" /workspace/research/vllm-serving/scripts/vllm_bf16_entry.py "${args[@]}" \
  2>&1 | tee "$output/server-$port.log"
