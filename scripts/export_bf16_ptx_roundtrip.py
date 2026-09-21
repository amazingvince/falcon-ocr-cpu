#!/usr/bin/env python3
"""One unmodified pinned PTX assembly/launch control; no observer candidate."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess

import torch
from safetensors.torch import load_file, save_file

from cuda_ptx_reference import DriverModule, elf_sections
from export_bf16_fused_substages import bits_equal
from fetch_reference import sha256
from reference_preflight import preflight

CONTROL_REPORT_SHA = "99f612c1d6b78405d5e0cd6c783a05f120fb44d09af98d9c0f3371f36b73f047"
PTXAS_SHA = "daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2"
MASK_NAMES = ["kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
              "img_mask", "img_indices", "sequence_indices", "non_pad_mask_id"]


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def launch(module, lse_module, q, k, v, mask, shared):
    raw = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
    log2 = torch.empty((1, 16, 144), dtype=torch.float32, device=q.device)
    maximum = torch.empty_like(log2)
    stream = torch.cuda.current_stream().cuda_stream
    module.launch_attention([q, k, v, log2, maximum, *mask, raw], stream, shared)
    natural = torch.empty_like(log2)
    lse_module.launch_natural_lse(log2, natural, stream)
    return {"raw": raw, "lse": natural, "log2_lse": log2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-ptx-roundtrip-v1"))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise ValueError("Preserve prior attempts; output already exists")
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    args.output.mkdir(parents=True)
    report = {"schema_version": 1, "environment": environment, "accepted": False,
              "status": "started", "assembler_invocations": 0, "observer_candidate": False,
              "cases": [], "checks": {}, "phase": "identity"}
    modules = []
    startup_hashes = {}
    input_hashes = {}
    def bind(path, expected=None):
        path = pathlib.Path(path)
        digest = sha256(path)
        require(expected is None or digest == expected, "Bound input changed: " + str(path))
        input_hashes[str(path)] = digest
        return digest

    def preserve(path, target):
        path = pathlib.Path(path)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == input_hashes[str(path)], "Input changed before snapshot: " + str(path))
        target.write_bytes(raw)
        input_hashes[str(target)] = hashlib.sha256(raw).hexdigest()

    def input_window():
        return {name: still_matches(pathlib.Path(name), digest) for name, digest in input_hashes.items()}

    def still_matches(path, digest):
        try:
            return sha256(path) == digest
        except OSError:
            return False
    try:
        control = root / "artifacts/reference/bf16-fused-substages-v5"
        bind(control / "report.json", CONTROL_REPORT_SHA)
        prior = json.loads((control / "report.json").read_text(encoding="utf-8"))
        require(all(case["checks"][key]["equal"] for case in prior["cases"]
                    for key in ["raw_original_vs_frozen", "lse_original_vs_frozen"]), "Original control was not frozen-exact")
        for name, digest in prior["sources"].items():
            bind(control / name, digest)
        bind(control / "capture.safetensors", prior["capture_sha256"])
        bind(control / "mask-inputs.safetensors", prior["mask_inputs_sha256"])
        fixture = root / "artifacts/reference/attention-operators-bf16.safetensors"
        bind(fixture, prior["fixture_sha256"])
        for label in ["original", "original.natural_lse"]:
            for artifact in prior["compiled_kernels"][label][0]["artifacts"].values():
                path = root / artifact["path"]
                bind(path, artifact["sha256"])
            metadata_identity = prior["compiled_kernels"][label][0]["selected_launcher"]
            metadata_name = "original-0.metadata.json" if label == "original" else "original-natural-lse-0.metadata.json"
            bind(control / metadata_name, metadata_identity["metadata_sha256"])
        metadata = prior["compiled_kernels"]["original"][0]["metadata"]
        require(metadata["target"] == {"backend": "cuda", "arch": 89, "warp_size": 32}, "Unexpected target")
        require(metadata["num_warps"] == 4 and metadata["num_stages"] == 3 and metadata["shared"] == 27136,
                "Pinned launch metadata changed")
        require(metadata["enable_fp_fusion"] is True and metadata["ptx_options"] is None
                and metadata["maxnreg"] is None and metadata["global_scratch_size"] == 0
                and metadata["profile_scratch_size"] == 0, "Unexpected compiler or scratch settings")
        from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path
        _set_triton_ptxas_path()
        from triton.backends.nvidia.compiler import get_ptxas
        from triton import knobs
        assembler = pathlib.Path(get_ptxas(89).path)
        bind(assembler, PTXAS_SHA)
        require(not knobs.compilation.disable_line_info and not knobs.nvidia.disable_ptxas_opt,
                "Actual assembly settings differ from original normal line-info/optimization path")
        sources = ["scripts/export_bf16_ptx_roundtrip.py", "scripts/cuda_ptx_reference.py",
                   "scripts/export_bf16_fused_substages.py", "scripts/export_reference.py",
                   "scripts/instrument_bf16_flex.py", "scripts/verify_bf16_reduction_lowering.py",
                   "scripts/reference_preflight.py", "scripts/fetch_reference.py", "scripts/run_reference.sh",
                   "requirements/reference-lock.txt", "reference/manifest.json", "artifacts/model/artifact-manifest.json"]
        source_dir = args.output / "sources"
        source_dir.mkdir()
        for name in sources:
            path = root / name
            raw = path.read_bytes()
            startup_hashes[name] = hashlib.sha256(raw).hexdigest()
            archived = source_dir / name.replace("/", "__")
            archived.write_bytes(raw)
            input_hashes[str(archived)] = startup_hashes[name]
        for name in ["original.py", "original-0.ptx", "original-0.cubin", "original-0.metadata.json", "report.json",
                     "original-natural-lse-0.ptx", "original-natural-lse-0.cubin", "original-natural-lse-0.metadata.json"]:
            preserve(control / name, source_dir / ("preserved-" + name))
        for path, name in [(fixture, "fixture.safetensors"), (control / "mask-inputs.safetensors", "mask-inputs.safetensors"),
                           (control / "capture.safetensors", "control-capture.safetensors")]:
            preserve(path, source_dir / name)
        from torch._inductor.runtime import compile_tasks
        from triton.backends.nvidia import compiler as nvidia_compiler
        for module, name in [(compile_tasks, "installed-torch-compile-tasks.py"), (nvidia_compiler, "installed-triton-nvidia-compiler.py")]:
            path = pathlib.Path(module.__file__)
            bind(path)
            preserve(path, source_dir / name)
        report.update(source_sha256=startup_hashes, control_report_sha256=CONTROL_REPORT_SHA,
                      fixture_sha256=sha256(fixture), mask_inputs_sha256=prior["mask_inputs_sha256"],
                      control_capture_sha256=prior["capture_sha256"], original_metadata=metadata,
                      input_sha256=input_hashes.copy(), assembler={"path": str(assembler), "sha256": sha256(assembler),
                                 "version": subprocess.check_output([str(assembler), "--version"], text=True),
                                 "TRITON_PTXAS_PATH": os.environ.get("TRITON_PTXAS_PATH"),
                                 "disable_line_info": knobs.compilation.disable_line_info,
                                 "disable_ptxas_opt": knobs.nvidia.disable_ptxas_opt},
                      launch={"grid": [2, 1, 16], "block": [128, 1, 1], "shared_bytes": 27136,
                              "argument_types": ["device_pointer"] * 14 + ["u32"] * 5 + ["null_scratch_pointer"] * 2,
                              "u32_values": [144, 16384, 2, 2, 2]})
        (args.output / "startup.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        ptx = args.output / "original.ptx"
        ptx.write_bytes((source_dir / "preserved-original-0.ptx").read_bytes())
        bind(ptx, input_hashes[str(control / "original-0.ptx")])
        text = ptx.read_text(encoding="utf-8")
        header = text.split(".visible .entry", 1)[1].split(")", 1)[0]
        types = re.findall(r"\.param \.(u64|u32)\b", header)
        require(types == ["u64"] * 14 + ["u32"] * 5 + ["u64"] * 2, "Saved PTX ABI differs")
        require(".version 9.0" in text and ".target sm_89" in text and ".reqntid 128" in text, "PTX target/block differs")
        assembled = args.output / "roundtrip.cubin"
        command = [str(assembler), "-lineinfo", "-v", "--gpu-name=sm_89", str(ptx), "-o", str(assembled)]
        report.update(phase="single_assembly", assembler_invocations=1, assembler_command=command)
        require(all(input_window().values()), "Bound inputs changed before assembly")
        completed = subprocess.run(command, capture_output=True, text=True)
        (args.output / "ptxas.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (args.output / "ptxas.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, "The one permitted unmodified PTX assembly failed")
        section = ".text." + metadata["name"]
        original_code = elf_sections((source_dir / "preserved-original-0.cubin").read_bytes())[section]
        assembled_bytes = assembled.read_bytes()
        input_hashes[str(assembled)] = hashlib.sha256(assembled_bytes).hexdigest()
        rebuilt_code = elf_sections(assembled_bytes)[section]
        report["checks"]["entire_machine_text_section_exact"] = original_code == rebuilt_code
        report["machine_text"] = {"section": section, "bytes": len(original_code),
                                  "original_sha256": hashlib.sha256(original_code).hexdigest(),
                                  "roundtrip_sha256": hashlib.sha256(rebuilt_code).hexdigest()}
        disassembler = pathlib.Path("/usr/local/cuda/bin/nvdisasm")
        bind(disassembler)
        report["disassembler"] = {"path": str(disassembler), "sha256": sha256(disassembler),
                                  "version": subprocess.check_output([str(disassembler), "--version"], text=True)}
        for label, cubin in [("original", source_dir / "preserved-original-0.cubin"), ("roundtrip", assembled)]:
            output = subprocess.check_output([str(disassembler), "-ndf", "-g", str(cubin)])
            (args.output / (label + ".sass")).write_bytes(output)
        require(report["checks"]["entire_machine_text_section_exact"], "Machine code changed; no launch or assembler/flag sweep permitted")
        report["phase"] = "launch_abi_control"
        lse_metadata = prior["compiled_kernels"]["original.natural_lse"][0]["metadata"]
        lse_ptx = (source_dir / "preserved-original-natural-lse-0.ptx").read_text(encoding="utf-8")
        require(lse_metadata["shared"] == 0 and lse_metadata["num_warps"] == 4
                and ".reqntid 128" in lse_ptx and "shl.b32 \t%r1, %r3, 7;" in lse_ptx,
                "Preserved natural LSE launch shape differs")
        lse_types = re.findall(r"\.param \.(u64|u32)\b", lse_ptx.split(".visible .entry", 1)[1].split(")", 1)[0])
        require(lse_types == ["u64", "u64", "u64", "u32", "u64", "u64"], "Preserved natural LSE ABI differs")
        require(all(input_window().values()), "Bound inputs changed before launch")
        lse_cubin = source_dir / "preserved-original-natural-lse-0.cubin"
        lse_module = DriverModule(lse_cubin, lse_metadata["name"], input_hashes[str(lse_cubin)])
        modules.append(lse_module)
        report["natural_lse_launch"] = {"preserved_cubin_sha256": sha256(control / "original-natural-lse-0.cubin"),
                                        "grid": [18, 1, 1], "block": [128, 1, 1], "shared_bytes": 0,
                                        "arguments": ["input_pointer", "output_pointer", "u64(144)", "u32(2304)", "null_scratch", "null_scratch"]}
        frozen = load_file(str(source_dir / "fixture.safetensors"))
        prior_tensors = load_file(str(source_dir / "control-capture.safetensors"))
        mask_values = load_file(str(source_dir / "mask-inputs.safetensors"))
        mask = [mask_values["mask." + name].cuda() for name in MASK_NAMES]
        original_cubin = source_dir / "preserved-original-0.cubin"
        original_module = DriverModule(original_cubin, metadata["name"], input_hashes[str(original_cubin)])
        modules.append(original_module)
        roundtrip_module = DriverModule(assembled, metadata["name"], input_hashes[str(assembled)])
        modules.append(roundtrip_module)
        attributes = {label: {"max_threads": mod.attribute(0), "static_shared_bytes": mod.attribute(1),
                              "registers": mod.attribute(4), "ptx_version": mod.attribute(5), "binary_version": mod.attribute(6)}
                      for label, mod in [("original", original_module), ("roundtrip", roundtrip_module)]}
        report["function_attributes"] = attributes
        report["loaded_module_sha256"] = {"original": original_module.loaded_image_sha256,
                                          "roundtrip": roundtrip_module.loaded_image_sha256,
                                          "natural_lse": lse_module.loaded_image_sha256}
        report["checks"]["function_attributes_exact"] = attributes["original"] == attributes["roundtrip"]
        require(report["checks"]["function_attributes_exact"], "Loaded function attributes differ")
        runtime_paths = set()
        for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith("/"):
                path = pathlib.Path(parts[5])
                if path.name.startswith(("libcuda.so", "libcudart.so", "libc10_cuda.so", "libtorch_cuda.so")):
                    runtime_paths.add(path)
        require(any(path.name.startswith("libcuda.so") for path in runtime_paths), "Loaded CUDA Driver library identity unavailable")
        report["runtime_library_sha256"] = {str(path): bind(path) for path in sorted(runtime_paths)}
        report["input_sha256"] = input_hashes.copy()
        (args.output / "launch-startup.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        require(all(input_window().values()), "Bound inputs/runtime changed before kernel execution")
        tensors = {}
        with torch.inference_mode():
            for case in ["prefill.layer.0", "prefill.layer.17", "prefill.layer.19"]:
                q = torch.empty_strided((1, 16, 144, 64), (147456, 64, 1024, 1), dtype=torch.bfloat16, device="cuda:0")
                k = torch.empty_strided(q.shape, (262144, 16384, 64, 1), dtype=q.dtype, device=q.device)
                v = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
                for tensor, name in [(q, "q"), (k, "k"), (v, "v")]:
                    tensor.copy_(frozen[case + "." + name].cuda().transpose(0, 1).unsqueeze(0))
                original = launch(original_module, lse_module, q, k, v, mask, metadata["shared"])
                rebuilt = launch(roundtrip_module, lse_module, q, k, v, mask, metadata["shared"])
                torch.cuda.synchronize()
                checks = {}
                for name in ["raw", "lse", "log2_lse"]:
                    checks[name + ".roundtrip_vs_original"] = bits_equal(rebuilt[name], original[name])
                    checks[name + ".original_vs_saved_control"] = bits_equal(original[name], prior_tensors[case + ".original." + name])
                    for label, outputs in [("original", original), ("roundtrip", rebuilt)]:
                        tensors[case + "." + label + "." + name] = outputs[name].cpu().contiguous()
                checks["raw.original_vs_frozen"] = bits_equal(original["raw"], frozen[case + ".sink_free_expected"].transpose(0, 1).unsqueeze(0))
                checks["lse.original_vs_frozen"] = bits_equal(original["lse"], frozen[case + ".lse"].unsqueeze(0))
                case_report = {"case": case, "checks": checks, "accepted": all(value["equal"] for value in checks.values()),
                               "inputs": {name: {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}
                                          for name, value in [("q", q), ("k", k), ("v", v)]}}
                report["cases"].append(case_report)
        save_file(tensors, str(args.output / "outputs.safetensors"))
        require(all(case["accepted"] for case in report["cases"]), "Original-PTX roundtrip numerical or launch-ABI control failed")
        report.update(accepted=True, status="accepted_unmodified_ptx_roundtrip", phase="complete")
    except Exception as error:
        report.update(accepted=False, status="rejected_unmodified_ptx_roundtrip", error=repr(error))
    finally:
        cleanup_errors = []
        try:
            torch.cuda.synchronize()
        except Exception as error:
            cleanup_errors.append("synchronize: " + repr(error))
        for module in reversed(modules):
            try:
                module.close()
            except Exception as error:
                cleanup_errors.append("module unload: " + repr(error))
        if cleanup_errors:
            report.update(accepted=False, status="rejected_cuda_cleanup_error", cleanup_errors=cleanup_errors)
        report["source_window_unchanged"] = {name: still_matches(root / name, digest) for name, digest in startup_hashes.items()}
        report["input_sha256"] = input_hashes
        report["input_window_unchanged"] = input_window()
        if not all(report["source_window_unchanged"].values()) or not all(report["input_window_unchanged"].values()):
            report.update(accepted=False, status="rejected_source_window_change")
        report["artifacts"] = {path.relative_to(args.output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                               for path in args.output.rglob("*") if path.is_file() and path.name != "report.json"}
        report["qualification"] = "One unmodified PTX assembly and explicit Driver ABI control. No observer candidate or production changes; complete output and original frozen-control equality required. No assembler/flag sweep."
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "accepted", "phase", "assembler_invocations", "checks"]}, indent=2), flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
