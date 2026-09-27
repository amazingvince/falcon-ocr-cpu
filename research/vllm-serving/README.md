# vllm-serving

Production-style BF16 serving with vLLM and the held-out comparisons made through it.

| File | What it did or produced | Status |
|---|---|---|
| [scripts/audit_vllm_precision.py](scripts/audit_vllm_precision.py) | Inspect compiled Triton PTX from this fresh, single-request serving container | history |
| [scripts/capture_serving_harness.py](scripts/capture_serving_harness.py) | Preserve the full-page serving harness after launch, before HTTP inference | history |
| [scripts/compare_vllm_preprocessing.py](scripts/compare_vllm_preprocessing.py) | CPU-only comparison of pinned serving helper source and pinned HF preprocessing | history |
| [scripts/dump_vllm_runtime_sources.py](scripts/dump_vllm_runtime_sources.py) | Preserve the specific installed serving model/backend sources from its image | history |
| [scripts/finalize_vllm_reference.py](scripts/finalize_vllm_reference.py) | Attach post-request precision/source/log evidence to the durable serving report | history |
| [scripts/inspect_vllm_image.py](scripts/inspect_vllm_image.py) | Read immutable public OCI metadata; never pulls large image layers | history |
| [scripts/podman_reference.sh](scripts/podman_reference.sh) | Project-specific rootful Podman storage. This WSL VHD is backed by D | history |
| [scripts/prepare_serving_fullpages.py](scripts/prepare_serving_fullpages.py) | Freeze a bounded serving sample from visual evidence, before serving outputs | history |
| [scripts/request_vllm_draftgen.py](scripts/request_vllm_draftgen.py) | Greedy full-page transcripts of a page manifest for draft-head training (concurrent, resumable) | active |
| [scripts/request_vllm_fullpages.py](scripts/request_vllm_fullpages.py) | Run bounded natural full-page HTTP requests against the pinned vLLM fork | history |
| [scripts/request_vllm_heldout.py](scripts/request_vllm_heldout.py) | Greedy full-page requests against a running vLLM server | history |
| [scripts/request_vllm_reference.py](scripts/request_vllm_reference.py) | Make and preserve an actual direct-vLLM multimodal serving smoke request | history |
| [scripts/run_reference_vllm.sh](scripts/run_reference_vllm.sh) | Isolated, digest-pinned vLLM only; the image's pipeline service is not started | history |
| [scripts/run_vllm_bf16.sh](scripts/run_vllm_bf16.sh) | Production-style BF16 vLLM server (the image entrypoint's vLLM flags with | history |
| [scripts/verify_serving_fullpages_evidence.py](scripts/verify_serving_fullpages_evidence.py) | Freeze file-identity checks for the completed three-page serving references | history |
| [scripts/vllm_bf16_entry.py](scripts/vllm_bf16_entry.py) | Verify the model assets, record the environment, then start vLLM's OpenAI server | history |
| [scripts/vllm_reference_entry.py](scripts/vllm_reference_entry.py) | Gate the immutable serving container before its real OpenAI API entrypoint | history |
