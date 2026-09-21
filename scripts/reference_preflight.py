#!/usr/bin/env python3
"""Repository-specific GPU identity, immutable input, and environment launch gate."""
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

from fetch_reference import REVISION, WEIGHT_SHA256, sha256

UUID = "GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58"
EXPECTED = {"torch": "2.11.0+cu130", "transformers": "5.14.1", "tokenizers": "0.22.2",
            "safetensors": "0.8.0", "einops": "0.8.1", "pillow": "12.3.0", "numpy": "2.5.1"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight(root):
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "cuBLAS workspace configuration must be set to :4096:8 before CUDA initialization")
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == UUID, "GPU must be isolated by full UUID")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.free,driver_version",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    matches = [row.split(", ") for row in rows if row.startswith(UUID + ",")]
    require(len(matches) == 1 and matches[0][1] == "NVIDIA GeForce RTX 4090", "NVIDIA UUID/model identity mismatch: " + repr(matches))
    required = int(os.environ.get("FOCR_MIN_FREE_MIB", "4096"))
    require(int(matches[0][2]) >= required, f"Need {required} MiB free; got {matches[0][2]}")
    require(torch.cuda.device_count() == 1, "Expected exactly one logical CUDA device")
    properties = torch.cuda.get_device_properties(0)
    require(properties.name == "NVIDIA GeForce RTX 4090", "Torch device model differs from the expected RTX 4090")
    actual_uuid = str(properties.uuid)
    normalized_uuid = actual_uuid.removeprefix("GPU-").lower()
    require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", normalized_uuid) is not None,
            "Torch returned an unsupported device UUID representation: " + actual_uuid)
    require(normalized_uuid == UUID.removeprefix("GPU-").lower(), "Actual Torch CUDA device UUID differs from the selected physical GPU")
    torch.empty(1, device="cuda:0")
    actual = {name: importlib.metadata.version(name) for name in EXPECTED}
    require(actual == EXPECTED, "Pinned package versions differ: " + repr({"actual": actual, "expected": EXPECTED}))
    require(platform.python_version() == "3.12.13", "Python version differs: " + platform.python_version())
    # Validate every resolved dependency, not only the top-level requirements.
    for line in (root / "requirements/reference-lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            require(importlib.metadata.version(name) == version, f"Dependency mismatch: {name}")
    artifact = json.loads((root / "artifacts/model/artifact-manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "reference/manifest.json").read_text(encoding="utf-8"))
    require(artifact["model_revision"] == manifest["model_revision"] == REVISION and
            artifact["weights_sha256"] == manifest["weights_sha256"] == WEIGHT_SHA256, "Model revision or checkpoint pin differs")
    require(artifact["files"] == manifest["files"], "Artifact digest/size set differs from the durable model manifest")
    for filename, info in artifact["files"].items():
        asset = root / "artifacts/model" / filename
        require(asset.stat().st_size == info["bytes"] and sha256(asset) == info["sha256"], "Model asset differs: " + filename)
    require(artifact["files"]["model.safetensors"]["sha256"] == WEIGHT_SHA256, "Checkpoint digest differs")
    return {"gpu_uuid": UUID, "gpu_name": matches[0][1], "free_memory_mib": int(matches[0][2]),
            "torch_device_uuid": actual_uuid,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "driver": matches[0][3], "cuda_runtime": torch.version.cuda, "packages": actual,
            "python": platform.python_version(), "platform": platform.platform(),
            "environment": sys.prefix, "python_executable": sys.executable, "wsl": True,
            "requirements_sha256": sha256(root / "requirements/reference-lock.txt")}


if __name__ == "__main__":
    print(json.dumps(preflight(pathlib.Path(__file__).resolve().parents[1]), indent=2))
