#!/usr/bin/env python3
"""Bind future corpus references to immutable source/runtime identity before inference.

This module does not retrofit older runs. Its archive is created before the first
model forward and must already exist with identical content when resuming.
"""
import datetime
import hashlib
import json
import os
import pathlib

from fetch_reference import REVISION, WEIGHT_SHA256, sha256

SOURCE_PATHS = (
    "scripts/run_corpus_reference.py",
    "scripts/export_reference.py",
    "scripts/reference_preflight.py",
    "scripts/fetch_reference.py",
    "scripts/reference_run_identity.py",
    "scripts/reference_corpus_contract.py",
    "scripts/validate_gpu_reference_record.py",
    "scripts/run_reference.sh",
    "requirements/reference-lock.txt",
    "reference/manifest.json",
    "artifacts/model/artifact-manifest.json",
)
ENVIRONMENT_KEYS = (
    "gpu_uuid", "torch_device_uuid", "gpu_name", "driver", "cuda_runtime", "packages",
    "python", "platform", "requirements_sha256", "cublas_workspace_config",
)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def capture_reference_identity(root, environment, precision):
    """Read identity and source bytes without mutating any run output."""
    root = pathlib.Path(root)
    if precision not in ["fp32", "bf16"]:
        raise ValueError("Unsupported reference precision")
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config != ":4096:8" or environment.get("cublas_workspace_config") != workspace_config:
        raise ValueError("Actual cuBLAS workspace environment does not match the pinned numerical policy")
    contents = {name: (root / name).read_bytes() for name in SOURCE_PATHS}
    assets = json.loads(contents["artifacts/model/artifact-manifest.json"])
    manifest = json.loads(contents["reference/manifest.json"])
    if assets["model_revision"] != REVISION or assets["weights_sha256"] != WEIGHT_SHA256:
        raise ValueError("Artifact manifest does not identify the pinned reference model")
    if manifest["model_revision"] != REVISION or manifest["weights_sha256"] != WEIGHT_SHA256 or assets["files"] != manifest["files"]:
        raise ValueError("Artifact digest set differs from the durable reference model manifest")
    # Actual asset contents are checked by the mandatory preflight immediately
    # before this capture. Keep the entire expected digest/size set in identity.
    source_files = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    for name, data in contents.items()}
    policy = {
        "precision": precision,
        "parameter_cast": ("HF eager model-wide FP32, including golden frequencies; temporal complex64"
                           if precision == "fp32" else
                           "HF eager model-wide BF16, including golden frequencies, then regenerate temporal complex64"),
        "torch_float32_matmul_precision": "highest",
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
        "torch_cuda_allow_fp16_reduced_precision_reduction": False,
        "torch_cuda_allow_bf16_reduced_precision_reduction": False,
        "flex_float32_precision": "ieee",
        "compiled_transformer_blocks": False,
        "greedy_tie_rule": "torch.argmax: first vocabulary index among equal maxima",
        "teacher_forced": False,
        "text_decode": "Pinned Transformers BPE decode(skip_special_tokens=False), remove named EOS strings, then Python str.strip()",
        "stop_token_ids": [11, 263],
        "cublas_workspace_config": workspace_config,
    }
    identity = {"schema_version": 1, "source_files": source_files, "verified_model_assets": assets["files"],
                "model_revision": REVISION, "weights_sha256": WEIGHT_SHA256,
                "runtime": {key: environment[key] for key in ENVIRONMENT_KEYS}, "numerical_policy": policy}
    identity["identity_sha256"] = canonical_digest(identity)
    return identity, contents


def preserve_reference_identity(output, identity, contents, *, new_run):
    """Create a pre-inference archive, or verify an existing resume archive."""
    output = pathlib.Path(output)
    folder = output / "provenance"
    manifest_path = folder / "startup-identity.json"
    if set(contents) != set(SOURCE_PATHS) or set(identity.get("source_files", {})) != set(SOURCE_PATHS):
        raise ValueError("Source archive must contain exactly the declared reference source paths")
    expected_digest = canonical_digest({k: v for k, v in identity.items() if k != "identity_sha256"})
    if identity.get("identity_sha256") != expected_digest:
        raise ValueError("Runtime identity digest is invalid")
    if not new_run and not manifest_path.exists():
        raise ValueError("Resume lacks a startup identity archive; do not retrofit an older run")
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior["identity"] != identity:
            raise ValueError("Preserved startup identity changed")
    for name, data in contents.items():
        expected = identity["source_files"][name]
        if hashlib.sha256(data).hexdigest() != expected["sha256"] or len(data) != expected["bytes"]:
            raise ValueError("Source bytes differ from captured identity: " + name)
        target = folder / "sources" / name
        if target.exists():
            if sha256(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]:
                raise ValueError("Preserved source archive changed: " + name)
        else:
            if not new_run:
                raise ValueError("Resume source archive is incomplete: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
    if not manifest_path.exists():
        record = {"identity": identity, "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "capture_phase": "after mandatory model/package/device preflight, before model initialization and first forward",
                  "qualification": "Startup source/dependency/runtime identity for this new run. Existing historical runs are not retroactively attested."}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return {"startup_identity_path": manifest_path.as_posix(), "startup_identity_sha256": sha256(manifest_path)}
