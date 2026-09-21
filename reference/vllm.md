# Actual pinned serving reference

`vllm-smoke-fp32.json` records an actual direct-vLLM HTTP request. The canonical
image produced the same 144 prompt tokens, all 17 output IDs including stop 263,
and exact decoded text as the strict GPU fixture. This is a tiny serving smoke;
full corpus/tensor parity and performance are separate gates.

The image is pinned to
`ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd`.
Its 32 compressed layers total 14,252,730,273 bytes; the installed image is
22,796,032,100 bytes. Podman 4.9.3 and crun 1.14.1 use dedicated storage at
`/home/amazi/falcon-ocr-vllm`, inside the WSL ext4 VHD backed by D:. Container
layers and temporary downloads must not be placed on the nearly full C: drive.
Docker Desktop is not required or modified.

The immutable image provides custom vLLM 0.1.0, PyTorch 2.10.0+cu128, Triton 3.6.0,
Transformers 5.3.0, and Pillow 12.2.0. The launch mounts independently verified
v1.5 weights/config/tokenizer read-only. It selects FP32 and `TRITON_ATTN`, highest
PyTorch matmul precision, NVIDIA TF32 override off, and Triton's IEEE default.
The post-request audit found compiled attention PTX and verified no TF32
instructions. The image's model math was not patched.

The fork rejects UUID strings in `CUDA_VISIBLE_DEVICES`, and ignores its own
documented `VLLM_ATTENTION_BACKEND` environment variable. Each launch resolves an
ordinal from the recorded RTX 4090 UUID, then verifies NVML and PyTorch UUIDs,
name, device count, free memory, and allocation. It uses the supported
`--attention-backend` flag. `vllm-attempts.json` records the failed first startup
and the successful second attempt.

Reproduce after installing pinned Podman/crun packages and pulling the image:

```bash
bash scripts/run_reference_vllm.sh
# Another terminal with the pinned reference Python environment:
python scripts/request_vllm_reference.py
bash scripts/podman_reference.sh exec falcon-ocr-vllm-reference \
  python /workspace/scripts/dump_vllm_runtime_sources.py
bash scripts/podman_reference.sh exec falcon-ocr-vllm-reference \
  python /workspace/scripts/audit_vllm_precision.py
bash scripts/podman_reference.sh stop --time 20 falcon-ocr-vllm-reference
python scripts/finalize_vllm_reference.py
```

The API binds only to loopback port 18080; no layout service starts. Completed
request/response, environment, server log, source hashes, and precision audit are
in `artifacts/reference/vllm-smoke-fp32`. The project-owned container was stopped
after the request, releasing the RTX 4090 for later reference probes.

The bounded full-page serving check is frozen in `serving-fullpages-v1-lock.json`:
one natural single-column journal page, one table-heavy scientific page and one
two-column manual page. These were selected from visual evidence before the
serving outputs. Both the pinned official plain engine and direct-vLLM HTTP
server completed all three: all 4,881 output IDs, raw text, prefix lengths and
EOS stops match the strict GPU references exactly. See
`official-fullpages-fp32.json` and `vllm-fullpages-fp32.json`.

The first official harness attempt failed before any forward because it passed
unsupported constructor options; its source and failure are preserved in
`artifacts/reference/official-fullpages-fp32-4096`. The corrected harness uses
the upstream per-sequence image-size fields and hardcoded disabled HR cache;
successful evidence is in `official-fullpages-fp32-4096-v2`. No upstream model
math was changed. The vLLM full-page run preserved four compiled PTX files,
including the attention kernel, and verified IEEE default with no TF32
instructions. All API stop IDs were directly observed in returned logprobs.
Installed sources, request/response bytes and the complete server log are
preserved, and the project-owned server was stopped. This is three-page output
qualification, not full-corpus serving, hidden-tensor or performance acceptance.

This image's multimodal processor defaults to a 1024-pixel longest side and a
1,003,520-pixel budget. Those defaults differ from the 1536-pixel HF target and
its 10,035,200-pixel budget. The full-page launch uses the supported
`--hf-overrides` image configuration to select matching preprocessing, with the
original model config and weights mounted unchanged. Request reports retain the
override, prompt counts, individual output IDs, API stop handling and the exact
request/response hashes. Source files and compiled precision are audited again.

Commands used for the completed runs (existing evidence paths are protected):

```bash
bash scripts/run_reference.sh --official-fullpages \
  --output artifacts/reference/official-fullpages-fp32-4096-v2
bash scripts/run_reference_vllm.sh --full-pages
# Another terminal using the pinned reference environment:
python scripts/request_vllm_fullpages.py
bash scripts/podman_reference.sh exec falcon-ocr-vllm-reference \
  python /workspace/scripts/dump_vllm_runtime_sources.py
bash scripts/podman_reference.sh exec falcon-ocr-vllm-reference \
  python /workspace/scripts/audit_vllm_precision.py
bash scripts/podman_reference.sh stop --time 20 falcon-ocr-vllm-reference
python scripts/finalize_vllm_reference.py --full-pages
```
