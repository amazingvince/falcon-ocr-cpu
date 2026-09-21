#!/usr/bin/env python3
"""Observe a copied pinned Flex kernel; reject any complete-output bit difference.

This is a queued diagnostic, not a model runner or a numerical policy. It uses
full original matrices, sparse schedules, and fused dot accumulator arguments.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from export_reference import import_model
from fetch_reference import sha256
from instrument_bf16_flex import FIELDS, FINAL_FIELDS, SLOTS_PER_LOOP, build_instrumented
from reference_preflight import preflight
from verify_bf16_reduction_lowering import verify_lowering


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def closure_tensors(function):
    return dict(zip(function.__code__.co_freevars, [x.cell_contents for x in function.__closure__]))


def bits_equal(actual, expected):
    actual, expected = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"equal": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
                "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    differing = (actual.view(integer) != expected.view(integer)).reshape(-1).nonzero().flatten()
    return {"equal": not len(differing), "different_count": len(differing), "elements": actual.numel(),
            "first_different_indices": differing[:32].tolist(),
            "max_absolute": float((actual.float() - expected.float()).abs().max())}


def preserve_kernel(kernel, output, label):
    entries = []
    for index, launcher in enumerate(kernel.launchers):
        if hasattr(launcher, "bin"):
            binary = launcher.bin
            metadata = binary.metadata._asdict() if hasattr(binary.metadata, "_asdict") else vars(binary.metadata)
            assembly = binary.asm
            selection = {"launcher_kind": "triton_compiled_kernel"}
        else:
            # Torch 2.11's static launcher retains the selected kernel in its
            # compile result and records the exact cache hash on the launcher.
            # Do not disable static launching or recompile with different flags.
            matches = [result for result in kernel.compile_results if result.config == launcher.config]
            if len(matches) != 1:
                raise RuntimeError("Selected static compile result is ambiguous")
            binary = matches[0].kernel
            from torch._inductor.runtime.runtime_utils import triton_cache_dir, triton_hash_to_path_key
            selected_hash = triton_hash_to_path_key(binary.hash)
            if selected_hash != launcher.cache_hash:
                raise RuntimeError("Selected launcher and retained kernel hashes differ")
            # load_kernel deliberately clears cubin_path/cubin_raw after loading.
            # Recover the same path from the retained selected hash/name, exactly
            # as Torch's reload_cubin_path does, without mutating or recompiling it.
            device = matches[0].compile_meta.get("device", 0)
            cubin_path = pathlib.Path(triton_cache_dir(0 if device is None else device)) / selected_hash / (binary.name + ".cubin")
            if binary.cubin_path is not None and pathlib.Path(binary.cubin_path) != cubin_path:
                raise RuntimeError("Retained cubin path differs from selected cache identity")
            if cubin_path.parent.name != launcher.cache_hash:
                raise RuntimeError("Selected launcher and retained cubin cache identities differ")
            metadata_path = cubin_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assembly = {}
            for kind in ["ptx", "cubin", "ttir", "ttgir", "llir"]:
                path = cubin_path.with_suffix("." + kind)
                if path.is_file():
                    assembly[kind] = path.read_bytes() if kind == "cubin" else path.read_text(encoding="utf-8")
            preserved_metadata = output / f"{label}-{index}.metadata.json"
            preserved_metadata.write_bytes(metadata_path.read_bytes())
            selection = {"launcher_kind": "torch_static_triton_launcher", "cache_hash": launcher.cache_hash,
                         "retained_kernel_hash": binary.hash, "retained_kernel_name": binary.name,
                         "selected_cubin_path": str(cubin_path), "selected_cubin_sha256": sha256(cubin_path),
                         "metadata_sha256": sha256(preserved_metadata), "compile_result_class": type(matches[0]).__name__,
                         "config": str(matches[0].config), "n_regs": launcher.n_regs,
                         "n_spills": launcher.n_spills, "shared": launcher.shared}
        paths = {}
        for kind in ["ptx", "cubin", "ttir", "ttgir", "llir"]:
            if kind not in assembly:
                continue
            value = assembly[kind]
            path = output / f"{label}-{index}.{kind}"
            path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
            paths[kind] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        ptx = assembly.get("ptx", "")
        instructions = [line.strip() for line in ptx.splitlines()
                        if re.search(r"\b(mma\.sync|ex2\.|cvt\.rn\.bf16|fma\.rn\.f32|div\.|rcp\.|lg2\.|sub\.f32|mul\.f32)", line)]
        entries.append({"metadata": metadata, "selected_launcher": selection, "artifacts": paths,
                        "instruction_counts": dict(collections.Counter(line.split()[0] for line in instructions)),
                        "selected_instruction_sequence": instructions})
    if not entries or not all("ptx" in x["artifacts"] and "cubin" in x["artifacts"] for x in entries):
        raise RuntimeError("Actual compiled launch PTX/CUBIN evidence unavailable")
    return entries


def run_kernel(module, args, observations=None, final_observations=None):
    q = args[0]
    raw = torch.empty_strided(q.shape, q.stride(), dtype=torch.bfloat16, device=q.device)
    lse_log2 = torch.empty((1, 16, 144), dtype=torch.float32, device=q.device)
    maximum = torch.empty_like(lse_log2)
    call = [*args[:3], lse_log2, maximum, *args[3:], raw, 144, 16384, 2, 2, 2]
    if observations is not None:
        call.extend([observations, final_observations])
    stream = torch.cuda.current_stream().cuda_stream
    module.triton_tem_fused_flex_attention_0.run(*call, 2, 1, 16, stream=stream)
    lse = torch.empty_like(lse_log2)
    # Preserve the pinned compiled log2-to-natural-LSE pointwise operation too.
    module.triton_poi_fused_mul_1.run(lse_log2, lse, 144, 16 * 144, stream=stream)
    return raw, lse, lse_log2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-fused-substages-v5"))
    parser.add_argument("--specification", type=pathlib.Path, default=pathlib.Path("reference/bf16-fused-substage-export-spec-v3.json"))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.output.exists():
        raise ValueError("Diagnostic output already exists; preserve prior attempts")
    args.output.mkdir(parents=True)
    spec_path = root / args.specification
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    unavailable_fields = spec.get("unavailable_fields", [])
    if unavailable_fields not in [[], ["sum_unrounded_probabilities_f32"]]:
        raise ValueError("Unexpected partial-observation exclusions")
    fixture_path = root / spec["inputs"]["fixture"]
    if sha256(fixture_path) != spec["inputs"]["fixture_sha256"]:
        raise ValueError("Frozen attention fixture changed")
    token_trace_path = root / spec["inputs"]["token_trace"]
    if sha256(token_trace_path) != spec["inputs"]["token_trace_sha256"]:
        raise ValueError("Frozen token trace changed")
    original, candidate = build_instrumented((root / spec["source"]["path"]).read_bytes(), omit_debug_sum=bool(unavailable_fields))
    for label, text in [("original", original), ("instrumented", candidate)]:
        (args.output / f"{label}.py").write_text(text, encoding="utf-8")
    for name in ["export_bf16_fused_substages.py", "instrument_bf16_flex.py", "verify_bf16_reduction_lowering.py"]:
        (args.output / name).write_bytes((root / "scripts" / name).read_bytes())
    (args.output / "specification.json").write_bytes(spec_path.read_bytes())
    sources = {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}
    original_module = load_module(args.output / "original.py", "focr_original_flex_observation")
    instrumented_module = load_module(args.output / "instrumented.py", "focr_instrumented_flex_observation")
    fixtures = load_file(str(fixture_path))
    with safe_open(str(token_trace_path), framework="pt", device="cpu") as trace:
        tokens = trace.get_tensor("tokens")
    model = import_model(root / "artifacts/model")
    attention = __import__("pinned_falcon_ocr.attention", fromlist=["create_batch_attention_mask"])
    config = model.FalconOCRConfig.from_json_file(str(root / "artifacts/model/config.json"))
    from tokenizers import Tokenizer
    pad = Tokenizer.from_file(str(root / "artifacts/model/tokenizer.json")).token_to_id("<|pad|>")
    padded = torch.full((1, 256), pad, dtype=torch.long, device="cuda:0")
    padded[:, :144] = tokens.cuda()
    mask = attention.create_batch_attention_mask(padded, pad_token_id=pad, eos_token_id=config.eos_id,
                                                 soi_token_id=config.image_cls_token_id, eoi_token_id=config.img_end_id, max_len=256)
    mask.seq_lengths = (144, 144)
    image = closure_tensors(attention.get_image_prefix_mask_mod(padded, config.image_cls_token_id, config.img_end_id))
    document = closure_tensors(attention.get_document_mask_mod(padded, config.eos_id))
    nonpad = closure_tensors(attention.get_non_left_pad_mask_mod(padded, pad))
    mask_inputs = [mask.kv_num_blocks, mask.kv_indices, mask.full_kv_num_blocks, mask.full_kv_indices,
                   image["img_mask"], image["img_indices"], document["sequence_indices"], nonpad["non_pad_mask_id"]]
    names = ["kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices", "img_mask", "img_indices", "sequence_indices", "non_pad_mask_id"]
    tensors = {"mask." + name: value.cpu().contiguous() for name, value in zip(names, mask_inputs)}
    tensors["mask.tokens"] = tokens.contiguous()
    tensors["mask.padded_tokens"] = padded.cpu().contiguous()
    mask_input_identity = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                "raw_bytes_sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
                           for name, value in tensors.items()}
    mask_input_path = args.output / "mask-inputs.safetensors"
    save_file(tensors, str(mask_input_path))
    reports = []
    with torch.inference_mode():
        for case in spec["inputs"]["cases"]:
            q = torch.empty_strided((1, 16, 144, 64), (147456, 64, 1024, 1), dtype=torch.bfloat16, device="cuda:0")
            k = torch.empty_strided((1, 16, 144, 64), (262144, 16384, 64, 1), dtype=torch.bfloat16, device="cuda:0")
            # Retain the original padded KV allocation strides explicitly.
            v = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
            for target, key in [(q, "q"), (k, "k"), (v, "v")]:
                target.copy_(fixtures[case + "." + key].cuda().transpose(0, 1).unsqueeze(0))
            observations = torch.full((16, 2, SLOTS_PER_LOOP, len(FIELDS), 64), torch.nan, device="cuda:0")
            final_observations = torch.full((16, len(FINAL_FIELDS), 64), torch.nan, device="cuda:0")
            raw, lse, log2 = run_kernel(original_module, [q, k, v, *mask_inputs])
            instrumented_raw, instrumented_lse, instrumented_log2 = run_kernel(instrumented_module, [q, k, v, *mask_inputs], observations, final_observations)
            torch.cuda.synchronize()
            frozen_raw = fixtures[case + ".sink_free_expected"].transpose(0, 1).unsqueeze(0)
            frozen_lse = fixtures[case + ".lse"].unsqueeze(0)
            checks = {"raw_instrumented_vs_original": bits_equal(instrumented_raw, raw),
                      "lse_instrumented_vs_original": bits_equal(instrumented_lse, lse),
                      "log2_lse_instrumented_vs_original": bits_equal(instrumented_log2, log2),
                      "raw_original_vs_frozen": bits_equal(raw, frozen_raw),
                      "lse_original_vs_frozen": bits_equal(lse, frozen_lse)}
            for label, value in [("original.raw", raw), ("original.lse", lse), ("original.log2_lse", log2),
                                 ("instrumented.raw", instrumented_raw), ("instrumented.lse", instrumented_lse), ("instrumented.log2_lse", instrumented_log2)]:
                tensors[case + "." + label] = value.cpu().contiguous()
            schedule = []
            observation_checks = []
            observations = observations.cpu()
            final_observations = final_observations.cpu()
            for probe in [p for p in spec["inputs"]["probes"] if p["case"] == case]:
                head, row = probe["head"], probe["query_row"]
                complete_stores = True
                for index, name in enumerate(FINAL_FIELDS):
                    value = final_observations[head, index]
                    stored = (value[:1] if index == 1 else value).contiguous().clone()
                    complete_stores &= not bool(torch.isnan(stored).any())
                    tensors[f"{case}.row{row}.head{head}.final.{name}"] = stored
                expected_schedule, observed_schedule = [], []
                query_block = row // 128
                for kind in range(2):
                    counts = mask.full_kv_num_blocks if kind else mask.kv_num_blocks
                    indices = mask.full_kv_indices if kind else mask.kv_indices
                    count = int(counts[0, 0, query_block])
                    starts = [int(block) * 128 + offset for block in indices[0, 0, query_block, :count].tolist() for offset in [0, 64]]
                    starts = starts[:max((q.shape[2] + 63) // 64, 1)]
                    expected_schedule.extend((kind, ordinal, start) for ordinal, start in enumerate(starts))
                    for ordinal in range(SLOTS_PER_LOOP):
                        values = observations[head, kind, ordinal]
                        key_start = values[FIELDS.index("key_start"), 0]
                        if torch.isnan(key_start):
                            continue
                        observed_schedule.append((kind, ordinal, int(key_start)))
                        label = f"{case}.row{row}.head{head}.{'full' if kind else 'partial'}{ordinal}"
                        schedule.append({"probe": probe, "loop_kind": kind, "loop_ordinal": ordinal, "key_start": int(key_start)})
                        for index, name in enumerate(FIELDS):
                            if name in unavailable_fields:
                                continue
                            vector = values[index] if index < 9 or name == "validity_mask" else values[index, :1]
                            complete_stores &= not bool(torch.isnan(vector).any())
                            tensors[label + "." + name] = vector.contiguous().clone()
                cast_matches_output = bits_equal(final_observations[head, 3].to(torch.bfloat16), instrumented_raw[0, head, row])
                observation_checks.append({"probe": probe, "all_requested_stores_present": complete_stores,
                                           "expected_schedule": expected_schedule, "observed_schedule": observed_schedule,
                                           "final_cast_matches_instrumented_output": cast_matches_output,
                                           "equal": complete_stores and expected_schedule == observed_schedule and cast_matches_output["equal"]})
            checks["observation_stores"] = {"equal": all(item["equal"] for item in observation_checks), "probes": observation_checks}
            reports.append({"case": case, "checks": checks, "accepted": all(x["equal"] for x in checks.values()),
                            "inputs": {name: {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)} for name, t in [("q", q), ("k", k), ("v", v)]},
                            "observed_schedule": schedule})
    target = args.output / "capture.safetensors"
    save_file(tensors, str(target))
    kernels, launch_configuration = {}, {}
    for label, module in [("original", original_module), ("instrumented", instrumented_module)]:
        attention_kernel = module.triton_tem_fused_flex_attention_0
        kernels[label] = preserve_kernel(attention_kernel, args.output, label)
        kernels[label + ".natural_lse"] = preserve_kernel(module.triton_poi_fused_mul_1, args.output, label + "-natural-lse")
        launch_configuration[label] = attention_kernel.inductor_meta.get("config_args")
        if not launch_configuration[label]:
            raise RuntimeError("Actual generated attention launch configuration is unavailable")
    lowering = verify_lowering(args.output)
    report = {"schema_version": 1, "environment": environment, "sources": sources, "fixture_sha256": sha256(fixture_path),
              "token_trace_sha256": sha256(token_trace_path), "mask_inputs_sha256": sha256(mask_input_path),
              "mask_input_tensors": mask_input_identity,
              "capture_sha256": sha256(target), "grid": [2, 1, 16], "launch_args": {"ks0": 144, "ks1": 16384, "ks2": 2, "ks3": 2, "ks4": 2},
              "SM_SCALE": 0.125, "RCP_LN2_bits": "3fb8aa3b", "cases": reports, "compiled_kernels": kernels,
              "generated_launch_configuration": launch_configuration,
              "unavailable_fields": unavailable_fields, "observation_scope": spec.get("observation_scope", "v2 full requested observation scope"),
              "reduction_lowering": lowering,
              "accepted": all(x["accepted"] for x in reports) and lowering["equal"],
              "qualification": "Observation-store candidate; interpret intermediates only when all complete raw BF16, F32 LSE, and frozen-fixture bit checks pass. Equality is necessary and does not prove identical internal instruction scheduling. No policy or production change."}
    report["status"] = ("accepted_partial_observation_candidate" if unavailable_fields else "accepted_observation_candidate") if report["accepted"] else "rejected_instrumentation_fixture_or_lowering_difference"
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": reports, "output": str(args.output)}, indent=2), flush=True)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
