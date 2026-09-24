#!/usr/bin/env bash
# Production-style BF16 vLLM server (the image entrypoint's vLLM flags with
# DTYPE=bfloat16), the full-page image config, top-32 logprobs, on the RTX 4090.
#   bash research/vllm-serving/scripts/run_vllm_bf16.sh <output dir>
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:?output directory}
uuid=GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58
image=ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd
[[ $(nvidia-smi -i "$uuid" --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 4090' ]]
physical_index=$(nvidia-smi -i "$uuid" --query-gpu=index --format=csv,noheader,nounits)
mkdir -p "$output"
args=(--model /models/Falcon-OCR --served-model-name falcon-ocr --host 127.0.0.1 --port 18081
  --max-model-len 16384 --max-num-seqs 2048 --gpu-memory-utilization 0.85
  --trust-remote-code --dtype bfloat16
  --chat-template /app/falcon_ocr_chat_template.jinja --limit-mm-per-prompt '{"image":1}'
  --no-enable-chunked-prefill --max-logprobs 32
  --hf-overrides '{"image_config":{"spatial_patch_size":16,"merge_size":1,"min_image_size":64,"max_image_size":1536,"min_pixels":3136,"max_pixels":10035200}}')
bash "$root/scripts/podman_reference.sh" run --rm --pull=never \
  --name falcon-ocr-vllm-bf16 --network host --device /dev/dxg \
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
  --entrypoint python "$image" /workspace/scripts/vllm_bf16_entry.py "${args[@]}" \
  2>&1 | tee "$output/server.log"
