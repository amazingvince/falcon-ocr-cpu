"""Read installed Torch identity/headers and exact-revision RMSNorm sources.

No CUDA initialization, tensor allocation, model import or kernel execution.
Run with the pinned reference Python and a fresh --output directory.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
import urllib.request

import torch

REVISION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
SOURCES = (
    "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
    "aten/src/ATen/native/cuda/thread_constants.h",
    "aten/src/ATen/native/cuda/block_reduce.cuh",
    "aten/src/ATen/native/layer_norm.cpp",
    "c10/cuda/CUDAMathCompat.h",
    "aten/src/ATen/AccumulateType.h",
)
HEADERS = (
    "include/c10/cuda/CUDAMathCompat.h",
    "include/ATen/native/cuda/thread_constants.h",
    "include/ATen/native/cuda/block_reduce.cuh",
    "include/ATen/AccumulateType.h",
    "version.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if torch.__version__ != "2.11.0+cu130" or torch.version.git_version != REVISION:
        raise RuntimeError("Installed Torch identity differs from the pinned source revision")
    if torch.cuda.is_initialized():
        raise RuntimeError("Source-only process unexpectedly has initialized CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    root = pathlib.Path(__file__).resolve().parents[2]
    torch_root = pathlib.Path(torch.__file__).resolve().parent
    report = {
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_executable": sys.executable, "torch_root": str(torch_root),
        "torch_version": torch.__version__, "torch_git_revision": torch.version.git_version,
        "cuda_build_version": torch.version.cuda, "torch_build_configuration": torch.__config__.show(),
        "cuda_initialized_before": False, "gpu_work": False,
        "disassembly_tools_on_path": {key: shutil.which(key) for key in ("cuobjdump", "nvdisasm", "readelf")},
        "installed": {}, "upstream": {}, "diagnostic": {},
    }

    def preserve(relative, data):
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return {"captured_path": relative, "bytes": len(data), "sha256": digest(data)}

    for name in HEADERS:
        data = (torch_root / name).read_bytes()
        report["installed"][name] = preserve("installed/" + name, data)
    for name in SOURCES:
        url = "https://raw.githubusercontent.com/pytorch/pytorch/" + REVISION + "/" + name
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        report["upstream"][name] = dict(preserve("upstream/" + name, data), url=url)
    for name in ("experiments/rms_norm/rstd_probe.rs", "experiments/rms_norm/capture_cuda_source_identity.py"):
        report["diagnostic"][name] = preserve("diagnostic/" + name, (root / name).read_bytes())
    report["cuda_initialized_after"] = torch.cuda.is_initialized()
    if report["cuda_initialized_after"]:
        raise RuntimeError("Source capture unexpectedly initialized CUDA")
    report["qualification"] = "Installed version and packaged-header identity plus source at the declared exact git revision. No device binary disassembly, kernel launch or proof of compiler lowering."
    output = args.output / "identity.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"identity": str(output), "sha256": digest(output.read_bytes()), "cuda_initialized": False}))


if __name__ == "__main__":
    main()
