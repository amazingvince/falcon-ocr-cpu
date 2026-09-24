#!/usr/bin/env python3
"""Gate the immutable serving container before its real OpenAI API entrypoint."""
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import runpy
import sys

import torch

MODEL_REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
WEIGHTS_SHA256 = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"
GPU_UUID = "GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58"
IMAGE = "ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_digest(stream)


def stream_digest(stream):
    state = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        state.update(chunk)
    return state.hexdigest()


def main():
    # This pinned vLLM fork casts CUDA_VISIBLE_DEVICES entries to int. Resolve
    # its ordinal from the UUID at launch, then independently verify both APIs.
    assert os.environ["FOCR_EXPECTED_GPU_UUID"] == GPU_UUID
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["FOCR_PHYSICAL_GPU_INDEX"]
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
    assert os.environ["VLLM_FLOAT32_MATMUL_PRECISION"] == "highest"
    model = pathlib.Path("/models/Falcon-OCR")
    assets = json.loads((model / "artifact-manifest.json").read_text())
    assert assets["model_revision"] == MODEL_REVISION
    assert assets["weights_sha256"] == WEIGHTS_SHA256
    verified_files = {}
    for name, expected in assets["files"].items():
        actual = digest(model / name)
        assert actual == expected["sha256"], f"Model asset hash mismatch: {name}"
        assert (model / name).stat().st_size == expected["bytes"], name
        verified_files[name] = actual
    assert verified_files["model.safetensors"] == WEIGHTS_SHA256
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ["FOCR_PHYSICAL_GPU_INDEX"]))
    nvml_uuid = pynvml.nvmlDeviceGetUUID(nvml_handle)
    if isinstance(nvml_uuid, bytes):
        nvml_uuid = nvml_uuid.decode()
    assert nvml_uuid == GPU_UUID
    cuda_uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert cuda_uuid.removeprefix("GPU-") == GPU_UUID.removeprefix("GPU-")
    allocation = torch.empty(1, device="cuda:0")
    free, total = torch.cuda.mem_get_info()
    assert free >= 12 * 1024 ** 3, f"Insufficient free GPU memory: {free} bytes"
    record = {"image": IMAGE, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
              "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu_uuid_selector": GPU_UUID, "gpu_name": torch.cuda.get_device_name(0), "visible_devices": 1,
              "derived_physical_index": int(os.environ["FOCR_PHYSICAL_GPU_INDEX"]), "nvml_uuid": nvml_uuid, "torch_uuid": cuda_uuid,
              "gpu_free_bytes": free, "gpu_total_bytes": total, "allocation_verified": allocation.is_cuda,
              "packages": {name: importlib.metadata.version(name) for name in ["vllm", "torch", "triton", "transformers", "tokenizers", "pillow", "safetensors"]},
              "precision": "fp32", "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
              "environment_controls": {name: os.environ.get(name) for name in ["NVIDIA_TF32_OVERRIDE", "TRITON_F32_DEFAULT", "VLLM_FLOAT32_MATMUL_PRECISION", "VLLM_ATTENTION_BACKEND"]},
              "model_config_sha256": digest(model / "config.json"), "tokenizer_sha256": digest(model / "tokenizer.json"),
              "tokenizer_config_sha256": digest(model / "tokenizer_config.json"),
              "artifact_manifest_sha256": digest(model / "artifact-manifest.json"), "verified_model_files": verified_files,
              "chat_template_sha256": digest("/app/falcon_ocr_chat_template.jinja"),
              "entrypoint_sha256": digest(__file__), "server_arguments": sys.argv[1:],
              "qualification": "Preflight passes only; actual serving and attention precision still require observed runtime evidence."}
    pathlib.Path("/out/environment.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if sys.argv[1:] != ["--probe"]:
        sys.argv[0] = "vllm.entrypoints.openai.api_server"
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
