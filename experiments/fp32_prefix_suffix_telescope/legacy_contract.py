"""Prospective fixed contract. Importing performs no I/O or numerical work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIND = "fp32-crossover-downstream-v1"
UUID = "GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58"
OLD_EXPORT = "experiments/fp32_crossover/export_gpu.py"
OLD_EXPORT_SHA = "9ce5cc6f1356770e3c8cb600074134495d2e0d2924b81c9148ed7d228d672e2b"
PINS = {
    "old_plan": ("artifacts/diagnostics/fp32-crossover-v1/plan.json", "63890f3c3c2f229db6f3d20a4f3aba6b15bf9fa7c080b5005dcc0b6bad1e435c"),
    "old_gpu_report": ("artifacts/diagnostics/fp32-crossover-gpu-v1/report.json", "72d577e05c7eb91d9d32890ef8e789fa988cdd80397a3e5b48cb4969a1e398ed"),
    "old_gpu_tensors": ("artifacts/diagnostics/fp32-crossover-gpu-v1/tensors.safetensors", "de8b170d0d9b9688c40fe0848fc77f8a4a1f62ce0f10ba3f76a109f887c88fcf"),
    "old_cpu_report": ("artifacts/diagnostics/fp32-crossover-v1/rust/report.json", "7947b11589c9e708ec6b7386a16727b677204ce0c55f516867a3f5c93e374e4a"),
    "old_cpu_tensors": ("artifacts/diagnostics/fp32-crossover-v1/rust/tensors.safetensors", "7b7e80488ccabbfc4c4d106f09ffb904a6591839527b62ac4a9a11fc2f163224"),
    "old_cpu_execution": ("artifacts/diagnostics/fp32-crossover-v1/execution.json", "8e122d4df988da03bafca42278b6f443ff8d38c040b06af61a13b17732ce8f87"),
    "old_analysis": ("reference/fp32-crossover-decomposition-v1.json", "b3e262ecc47713c356d3c8926fe1a642a610337eb077372705ff30510e067735"),
    "old_review": ("reference/fp32-crossover-independent-review-v1.json", "0bcc22437d8875f7c9f2feb44e31018061d78b61200db9b3a0718c7c2ddfae3b"),
    "layer7_plan": ("artifacts/diagnostics/fp32-crossover-layer7-v1/plan.json", "832d98ec8ba2807935017d0eb7ec2b5dca9ecfec0680211a5b41313f8554f4d2"),
    "layer7_gpu_report": ("artifacts/diagnostics/fp32-crossover-layer7-gpu-v1/report.json", "c3ed7215c44f33463ce49435298ab9701e79c2eea83054c028b516fc14089529"),
    "layer7_gpu_tensors": ("artifacts/diagnostics/fp32-crossover-layer7-gpu-v1/tensors.safetensors", "23f81b36dc0deb13a0aca72b757d250e5f5d5451718c215503c441c05fb0776b"),
    "layer7_cpu_report": ("artifacts/diagnostics/fp32-crossover-layer7-v1/rust/report.json", "c2ac7336a88ee47ca4416895c9a5797e21f680db7df3ebe6136571605933dfe8"),
    "layer7_cpu_tensors": ("artifacts/diagnostics/fp32-crossover-layer7-v1/rust/tensors.safetensors", "4972d70e6ec5c6c744745282141abb7768dbea6314970affa49603cd955e102d"),
    "layer7_cpu_execution": ("artifacts/diagnostics/fp32-crossover-layer7-v1/execution.json", "fb663eaf8fa203bd301a42ba23ed2fa673465dfad582742d7c8914a82d962729"),
    "layer7_completion": ("reference/fp32-crossover-layer7-completion-v1.json", "bcb6f6db4a2b08974f6cbf80d04102619ac20896619c6a6aee1dc7fbff299a12"),
    "layer7_review": ("reference/fp32-crossover-layer7-independent-review-v1.json", "8ac6549c9533969ea30fbefffdc6bf588269cb0b4fe24543f66c8f24366f27d7"),
    "gpu_trace": ("artifacts/reference/smoke-fp32/trace.safetensors", "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4"),
    "cpu_trace": ("artifacts/cpu/smoke-trace-sinks-pairwise.safetensors", "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309"),
    "policy": ("reference/tolerances-smoke-fp32-v1.json", "8f6382a4386b15f76e6984679208a44d0fa644b9d55a21ce579bbb26e0bfbe1c"),
    "model_manifest": ("artifacts/model/artifact-manifest.json", "eea9640e8ab9feccd9358a6b3526e579b14dc4bfcd47ecd561853863952b594d"),
}
RUNTIME = {"precision": "fp32", "rows": 144, "dim": 768, "heads": 16,
           "kv_heads": 8, "head_dim": 64, "ffn_dim": 2304,
           "layer": 8, "next_layer": 9, "gpu_cache_capacity": 256,
           "tf32": False, "flex_float32_precision": "ieee", "compiled_blocks": False,
           "gpu_uuid": UUID, "omp_threads": 8, "mkl_threads": 8,
           "cublas_workspace_config": ":4096:8"}
STAGES = {"input": [144, 768],
          "layer.8.attention_norm": [144, 768], "layer.8.qkv": [144, 2048],
          "layer.8.q": [144, 16, 64], "layer.8.k": [144, 16, 64],
          "layer.8.v": [144, 16, 64], "layer.8.attention": [144, 1024],
          "layer.8.wo": [144, 768], "layer.8.attention_residual": [144, 768],
          "layer.8.ffn_norm": [144, 768], "layer.8.w13": [144, 4608],
          "layer.8.gate": [144, 2304], "layer.8.w2": [144, 768],
          "layer.8.hidden": [144, 768], "layer.9.attention_norm": [144, 768],
          "layer.9.qkv": [144, 2048], "layer.9.v": [144, 16, 64]}
CONTROLS = list(STAGES)
ORIGINAL_CONTROLS = ["layer.8." + s for s in ("q", "k", "v", "attention", "hidden")] + ["layer.9.v"]
LAYER7_CONTROLS = ["layer.7." + s for s in ("q", "k", "v", "attention", "hidden")]
STATE_HASHES = {
    "G7": "165e72fa2186cf98816f04f6741545869d85630c75364965d50a53f0eb4155df",
    "C7": "890fb0045de1747a15e33b576535b3f2de489cde21358e6dbe76bef305db5dc5",
    "S": "846f58d046949cb2b5eaaccf4fc4c3bb99c57871bdd46f805516abc5f4a49ede"}
ENDPOINT = {"stage": "layer.9.v", "coordinate": [112, 14, 2],
            "original_absolute_bound": 0.005096435546875}
SOURCE_FILES = ["experiments/fp32_crossover_downstream/" + n for n in (
    "contract.py", "evidence.py", "native_segment.py", "source_guard.py", "prepare.py",
    "export_gpu.py", "compare.py", "test_source.py", "README.md")]
SOURCE_FILES += [OLD_EXPORT, "scripts/export_reference.py", "scripts/reference_preflight.py",
                 "scripts/fetch_reference.py", "requirements/reference-lock.txt", "reference/manifest.json",
                 "reference/fp32-crossover-downstream-next-v1.md"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(name):
    p = Path(name)
    require(not p.is_absolute() and "\\" not in name and ".." not in p.parts, "Expected relative POSIX path")
    p = (ROOT / p).resolve()
    require(p.is_relative_to(ROOT), "Path escaped project")
    return p


def read_bound(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Changed JSON: " + str(path))
    return json.loads(raw)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validate_shape_contract(plan):
    require(plan.get("kind") == KIND and plan.get("runtime") == RUNTIME, "Wrong downstream runtime")
    require(plan.get("stages") == STAGES and plan.get("control_stages") == CONTROLS, "Wrong stage/control inventory")
    require(plan.get("branches") == ["control_c", "substitution_s"] and plan.get("endpoint") == ENDPOINT,
            "Wrong ordered branch/endpoint scope")
    require(plan.get("inputs") == {k: {"path": p, "sha256": h} for k, (p, h) in PINS.items()}, "Changed evidence pins")
    require(plan.get("state_raw_sha256") == STATE_HASHES, "Wrong complete entry states")
    require(set(SOURCE_FILES) <= set(plan.get("source_sha256", {})), "Missing source closure")
    require(plan["source_sha256"][OLD_EXPORT] == OLD_EXPORT_SHA, "Old exporter source changed")
    require(plan.get("model_directory") == "artifacts/model", "Wrong model directory")


def plan_window(plan_path, expected):
    plan = read_bound(plan_path, expected)
    validate_shape_contract(plan)
    bound = {Path(plan_path).resolve(): expected}
    for item in plan["inputs"].values():
        bound[resolve(item["path"])] = item["sha256"]
    for name, digest in plan["source_sha256"].items():
        p = resolve(name)
        require(p not in bound or bound[p] == digest, "Source/input identity conflict")
        bound[p] = digest
    manifest = read_bound(resolve(PINS["model_manifest"][0]), PINS["model_manifest"][1])
    require(plan.get("model_files") == manifest["files"], "Model asset identity set differs")
    for name, record in manifest["files"].items():
        p = resolve("artifacts/model/" + name)
        require(p.stat().st_size == record["bytes"], "Wrong model asset size")
        require(p not in bound or bound[p] == record["sha256"], "Conflicting model/source digest")
        bound[p] = record["sha256"]
    archive = plan_path.parent / "source.zip"
    bound[archive] = plan["source_archive_sha256"]
    stable(bound)
    return plan, bound


def stable(bound):
    require(all(sha(p) == digest for p, digest in bound.items()), "Source/input/artifact changed")
