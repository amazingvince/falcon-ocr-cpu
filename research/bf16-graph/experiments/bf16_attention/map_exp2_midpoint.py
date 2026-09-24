"""CPU-only provenance join for one saved standalone exp2 BF16 midpoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bits(value):
    return f"{struct.unpack('<I',struct.pack('<f',value))[0]:08x}"


def main():
    prepared = ROOT / "artifacts/reference/bf16-exp2-arguments-v1"
    replay = ROOT / "artifacts/reference/bf16-exp2-gpu-replay-v1"
    manifest = json.loads((prepared / "manifest.json").read_bytes())
    result = json.loads((replay / "report.json").read_bytes())
    require(sha(prepared / "manifest.json") == result["argument_manifest_sha256"], "Argument identity differs")
    bindings = {str(prepared / "manifest.json"): sha(prepared / "manifest.json"), str(replay / "report.json"): sha(replay / "report.json")}
    for name, digest in manifest["outputs"].items():
        require(sha(prepared / name) == digest, "Prepared output changed")
        bindings[str(prepared / name)] = digest
    for name, digest in manifest["source_sha256"].items():
        path = Path(name)
        require(sha(path) == digest, "Frozen source input changed")
        bindings[str(path)] = digest
    for name, digest in result["compiled_artifacts"].items():
        path = replay / ("native-exp2." + name)
        require(sha(path) == digest, "Compiled replay artifact changed")
        bindings[str(path)] = digest
    for path, digest in [(replay / "outputs.safetensors", result["outputs_sha256"]),
                         (replay / "export_bf16_exp2_replay.py", result["script_sha256"]),
                         (ROOT / "research/bf16-graph/scripts/prepare_bf16_exp2_arguments.py", manifest["script_sha256"]),
                         (ROOT / "artifacts/bf16-attention-crossings/exp2-probe.exe", manifest["rust_binary_sha256"])]:
        require(sha(path) == digest, "Replay/preparation provenance changed")
        bindings[str(path)] = digest
    require(result["fused_evidence"]["native_intermediate_claim"] is False and not result["fused_same_argument_checks"], "Unexpected fused evidence")
    rows = result["rows"]
    require(len(rows) == 1876 and len({r['argument_bits'] for r in rows}) == 1876, "Replay inventory differs")
    require(all(r["native_ex2_bits"] == r["torch_exp2_bits"] and r["native_bf16_bits"] == r["torch_bf16_bits"] for r in rows), "Native/Torch disagreement")
    crossings = [r for r in rows if r["native_bf16_bits"] != r["rust_bf16_bits"]]
    require(len(crossings) == 1 and crossings[0]["argument_bits"] == "c0c9799f", "Crossing inventory differs")
    arguments = json.loads((prepared / "arguments.json").read_bytes())
    matches = []
    for index, record in enumerate(arguments["records"]):
        for lane, word in enumerate(record["argument_bits"]):
            if word == "c0c9799f":
                matches.append({"record_index_zero_based": index, "lane": lane, "absolute_key": record["tile"]["key_start"] + lane,
                                **{k:v for k,v in record.items() if k != "argument_bits"}})
    require(len(matches) == 1, "Midpoint origin is not unique")
    origin = matches[0]
    require(origin["backend"] == "avx512" and origin["condition"] == "cpu_score_gpu_max", "Origin differs")
    analysis_path = ROOT / "research/bf16-graph/experiments/bf16_attention/analyze_crossings.py"
    spec = importlib.util.spec_from_file_location("frozen_saved_tile_reader", analysis_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    gpu = reader.tensors(ROOT / "artifacts/reference/bf16-attention-tiles.safetensors")
    cpu = reader.tensors(ROOT / "artifacts/cpu/bf16-attention-tiles-avx512.safetensors")
    prefix = origin["probe"] + ".tile1"
    sources = {"gpu": gpu, "cpu": cpu}
    replay_rows = {r["argument_bits"]: r for r in rows}
    combinations = {}
    for score_source in ("gpu", "cpu"):
        for maximum_source in ("gpu", "cpu"):
            score = sources[score_source][prefix + ".scores_log2"][22]
            maximum = sources[maximum_source][prefix + ".maximum"][0]
            argument = np.float32(score - maximum)
            word = bits(argument)
            condition = score_source + "_score_" + maximum_source + "_max"
            saved = [r for r in arguments["records"] if r["backend"] == "avx512" and r["probe"] == origin["probe"] and r["tile"]["index"] == 1 and r["condition"] == condition]
            require(len(saved) == 1 and saved[0]["argument_bits"][22] == word, "Reconstructed saved argument differs")
            combinations[condition] = {"score":float(score), "score_bits":bits(score), "maximum":float(maximum), "maximum_bits":bits(maximum),
                                       "argument":float(argument), **replay_rows[word]}
    crossing = crossings[0]
    for prefix in ("native", "rust"):
        value = int(crossing[prefix + ("_ex2_bits" if prefix == "native" else "_exp2_bits")],16)
        rounded = (value + 0x7fff + ((value >> 16) & 1)) >> 16
        require(f"{rounded:04x}" == crossing[prefix + "_bf16_bits"], "RNE midpoint rounding differs")
    bindings[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    require(all(sha(Path(name)) == digest for name,digest in bindings.items()), "Analysis source window changed")
    report = {"schema_version":1,"status":"single_crossing_joined_to_saved_counterfactual",
              "artifact_sha256":bindings,"origin":origin,"crossing":crossing,"same_lane_combinations":combinations,
              "native_vs_rust_f32_differences":sum(r['native_ex2_bits'] != r['rust_exp2_bits'] for r in rows),
              "interpretation":"The unique BF16 function crossing is the AVX512 CPU-score/GPU-oracle-max counterfactual. Actual saved GPU-oracle and CPU arguments at this lane are respectivelyc0c9799e andc0c979a0; native GPU and Rust exp2 agree exactly on both. The saved actual probability crossing therefore persists as an argument difference, not a standalone exp2 discrepancy at either actual argument. The old Rust-only counterfactual claim that GPU max alone restores GPU BF16 probability does not hold for native GPU exp2 at the mixed argument.",
              "native_fused_argument_claim":False,"rejected_fused_v4_used":False,"gpu_calls":False,"policy_or_production_changes":False,
              "limits":"Saved GPU tiles are an independent oracle, not accepted native fused intermediates. This one lane does not settle denominator/PV or other FP32 exp2 effects. Later accepted fused evidence must be evaluated separately."}
    output = ROOT / "reference/bf16-exp2-midpoint-origin-v1.json"
    with output.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"origin":origin,"report_sha256":sha(output)}))


if __name__ == "__main__":
    main()
