#!/usr/bin/env python3
"""Verify the model assets, record the environment, then start vLLM's OpenAI server.

Production-style BF16 serving: unlike vllm_reference_entry.py (the FP32,
TF32-off reference), this keeps the image's own precision defaults
(VLLM_FLOAT32_MATMUL_PRECISION=high, TF32 not overridden) and --dtype bfloat16.
"""
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
IMAGE = "ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd"


def digest(path):
    state = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            state.update(chunk)
    return state.hexdigest()


def main():
    model = pathlib.Path("/models/Falcon-OCR")
    assets = json.loads((model / "artifact-manifest.json").read_text())
    assert assets["model_revision"] == MODEL_REVISION
    assert assets["weights_sha256"] == WEIGHTS_SHA256
    for name, expected in assets["files"].items():
        assert digest(model / name) == expected["sha256"], f"Model asset hash mismatch: {name}"
    arguments = sys.argv[1:]
    assert arguments[arguments.index("--dtype") + 1] == "bfloat16"
    assert torch.cuda.device_count() == 1
    record = {"image": IMAGE, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
              "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(0), "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
              "packages": {name: importlib.metadata.version(name) for name in ["vllm", "torch", "triton", "transformers"]},
              "precision": "bf16",
              "environment_controls": {name: os.environ.get(name) for name in [
                  "NVIDIA_TF32_OVERRIDE", "VLLM_FLOAT32_MATMUL_PRECISION", "VLLM_ATTENTION_BACKEND"]},
              "entrypoint_sha256": digest(__file__), "server_arguments": arguments,
              "qualification": "Production-style BF16 serving settings (image entrypoint defaults, DTYPE=bfloat16)."}
    pathlib.Path("/out/environment.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
