"""Inspect saved tile evidence and CPU-only same-argument exp2 counterfactuals."""
import argparse
import hashlib
import json
import math
import pathlib
import struct
import subprocess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "artifacts/reference/bf16-attention-tiles.safetensors"
OPERANDS = ROOT / "artifacts/reference/attention-operators-bf16.safetensors"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def bf16(value):
    word = bits(value)
    return f32bits(((word + 0x7fff + ((word >> 16) & 1)) >> 16) << 16)


def tensors(path):
    raw = pathlib.Path(path).read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    out = {}
    for name, spec in header.items():
        if name == "__metadata__": continue
        start, end = spec["data_offsets"]
        data = raw[8 + length + start:8 + length + end]
        if spec["dtype"] == "F32":
            values = np.frombuffer(data, dtype="<f4")
        elif spec["dtype"] == "BF16":
            values = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            continue
        out[name] = values.reshape(spec["shape"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp2-binary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("Refusing to overwrite evidence")
    reference = tensors(REFERENCE)
    operands = tensors(OPERANDS)
    meta = json.loads(REFERENCE.with_suffix(".json").read_text())
    assert sha(REFERENCE) == meta["output_sha256"] == "1de7df7665034a58e2647c9ac840decad24e9aca11c81ff6d4e4101189440e23"
    assert sha(OPERANDS) == meta["fixture_sha256"] == "78abfc0eba2a276d047b300518a3a19d0a098ffc875c81f009569092c7a08462"
    candidate_paths = {backend: ROOT / f"artifacts/cpu/bf16-attention-tiles-{backend}.safetensors" for backend in ("avx512", "scalar")}
    candidates = {k: tensors(p) for k, p in candidate_paths.items()}
    for backend, path in candidate_paths.items():
        prior = json.loads((ROOT / f"reference/bf16-attention-tile-differences-{backend}.json").read_text())
        assert sha(path) == prior["candidate_sha256"]
    argument_bits = set()
    work = []
    for backend, candidate in candidates.items():
        for probe in meta["probes"]:
            previous = {"gpu": -math.inf, "cpu": -math.inf}
            anchors = {"gpu": [], "cpu": []}
            for tile in probe["tiles"]:
                prefix = probe["name"] + f".tile{tile['index']}"
                data = {"backend": backend, "probe": probe, "tile": tile, "prefix": prefix}
                for label, source in (("gpu", reference), ("cpu", candidate)):
                    scores = source[prefix + ".scores_log2"]
                    maximum = float(source[prefix + ".maximum"][0])
                    actual_max = max(previous[label], float(np.max(scores)))
                    assert bits(maximum) == bits(actual_max), (prefix, label, "online max mismatch")
                    if maximum > previous[label]:
                        anchors[label] = [tile["key_start"] + int(i) for i in np.flatnonzero(scores == maximum)]
                    elif maximum == previous[label]:
                        anchors[label] += [tile["key_start"] + int(i) for i in np.flatnonzero(scores == maximum)]
                    if label == "gpu":
                        assert bits(float(source[prefix + ".maximum_before"][0])) == bits(previous[label])
                    for lane in range(tile["valid_keys"]):
                        score = float(scores[lane])
                        if math.isfinite(score):
                            expected_score = np.float32(np.float32(source[prefix + ".qk"][lane] * np.float32(0.125)) * np.float32(1.44269504))
                            assert bits(expected_score) == bits(score), (prefix, label, "score scaling")
                    data[label + "_maximum"] = maximum
                    data[label + "_previous_maximum"] = previous[label]
                    data[label + "_max_keys"] = list(anchors[label])
                    previous[label] = maximum
                for score_label, score_source in (("gpu", reference), ("cpu", candidate)):
                    for max_label in ("gpu", "cpu"):
                        values = (score_source[prefix + ".scores_log2"] - np.float32(data[max_label + "_maximum"])).astype(np.float32)
                        key = score_label + "_score_" + max_label + "_max"
                        data[key] = [bits(v) for v in values]
                        argument_bits.update(data[key])
                work.append(data)
    command = [str(args.exp2_binary.resolve())]
    request = "".join(f"{word:08x}\n" for word in sorted(argument_bits))
    completed = subprocess.run(command, input=request, text=True, capture_output=True, check=True)
    exp2 = {}
    for line in completed.stdout.splitlines():
        arg, answer, rounded = (int(x, 16) for x in line.split())
        exp2[arg] = (f32bits(answer), f32bits(rounded << 16))
    assert set(exp2) == argument_bits
    records = []
    for item in work:
        backend, probe, tile, prefix = (item[k] for k in ("backend", "probe", "tile", "prefix"))
        candidate = candidates[backend]
        gp = reference[prefix + ".probabilities_bf16"]
        cp = candidate[prefix + ".probabilities_bf16"]
        valid = np.isfinite(reference[prefix + ".scores_log2"])
        cpu_reproduction = []
        gpu_same_argument_differences = []
        gpu_same_argument_cast_differences = []
        for lane in range(64):
            value, rounded = exp2[item["cpu_score_cpu_max"][lane]]
            if bits(value) != bits(candidate[prefix + ".exp2"][lane]) or bits(rounded) != bits(cp[lane]):
                cpu_reproduction.append(lane)
            gpu_value, gpu_rounded = exp2[item["gpu_score_gpu_max"][lane]]
            if bits(gpu_value) != bits(reference[prefix + ".exp2"][lane]):
                gpu_same_argument_differences.append(lane)
            if bits(gpu_rounded) != bits(gp[lane]):
                gpu_same_argument_cast_differences.append(lane)
        assert not cpu_reproduction, (backend, prefix, "new Rust exp2 failed to reproduce saved CPU", cpu_reproduction)
        crossings = []
        for lane in np.flatnonzero(cp != gp):
            lane = int(lane); key = tile["key_start"] + lane
            row, head, case = probe["query_row"], probe["head"], probe["case"]
            q, k = operands[case + ".q"][row, head], operands[case + ".k"][key, head]
            exact_dot = math.fsum(float(a) * float(b) for a, b in zip(q, k))
            midpoint = (float(cp[lane]) + float(gp[lane])) / 2
            counterfactuals = {}
            for condition in ("gpu_score_gpu_max", "cpu_score_cpu_max", "gpu_score_cpu_max", "cpu_score_gpu_max"):
                argument = item[condition][lane]
                value, rounded = exp2[argument]
                counterfactuals[condition] = {"argument": f32bits(argument), "argument_bits": f"{argument:08x}",
                    "rust_exp2": value, "rust_bf16": rounded, "exp2_minus_midpoint": value - midpoint,
                    "matches_saved_gpu_bf16": bits(rounded) == bits(gp[lane])}
            anchor_details = {}
            for label, source in (("gpu", reference), ("cpu", candidate)):
                details = []
                for anchor in item[label + "_max_keys"]:
                    anchor_tile = next(t for t in probe["tiles"] if t["key_start"] <= anchor < t["key_start"] + t["valid_keys"])
                    anchor_prefix = probe["name"] + f".tile{anchor_tile['index']}"
                    offset = anchor - anchor_tile["key_start"]
                    kv = operands[case + ".k"][anchor, head]
                    details.append({"key": anchor, "qk": float(source[anchor_prefix + ".qk"][offset]),
                                    "f64_qk": math.fsum(float(a) * float(b) for a, b in zip(q, kv)),
                                    "score_log2": float(source[anchor_prefix + ".scores_log2"][offset])})
                anchor_details[label] = details
            crossings.append({"lane": lane, "key": key, "f64_qk": exact_dot,
                "gpu_qk": float(reference[prefix + ".qk"][lane]), "cpu_qk": float(candidate[prefix + ".qk"][lane]),
                "gpu_score": float(reference[prefix + ".scores_log2"][lane]), "cpu_score": float(candidate[prefix + ".scores_log2"][lane]),
                "gpu_probability_f32": float(reference[prefix + ".exp2"][lane]), "cpu_probability_f32": float(candidate[prefix + ".exp2"][lane]),
                "gpu_probability_bf16": float(gp[lane]), "cpu_probability_bf16": float(cp[lane]),
                "bf16_midpoint": midpoint, "gpu_exp2_minus_midpoint": float(reference[prefix + ".exp2"][lane]) - midpoint,
                "cpu_exp2_minus_midpoint": float(candidate[prefix + ".exp2"][lane]) - midpoint,
                "maximum_anchors": anchor_details, "counterfactuals": counterfactuals})
        records.append({"backend": backend, "probe": probe["name"], "tile": tile,
            "online_max_reproduced_exactly": True, "scores_reproduced_exactly": True,
            "rust_exp2_reproduces_saved_cpu_exactly": True, "unmasked_probability_count": int(valid.sum()),
            "gpu_same_argument_rust_exp2_different_lanes": gpu_same_argument_differences,
            "gpu_same_argument_rust_bf16_different_lanes": gpu_same_argument_cast_differences,
            "gpu_maximum": item["gpu_maximum"], "cpu_maximum": item["cpu_maximum"],
            "gpu_max_keys": item["gpu_max_keys"], "cpu_max_keys": item["cpu_max_keys"],
            "crossings": crossings})
    result = {"schema_version": 1, "scope": "Saved independent GPU tile oracle plus CPU-only exp2 interventions; no fused GPU intermediate capture or kernel change",
        "gpu_execution": False, "bounds_changed": False, "discarded_qk_candidates_retested": False,
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [REFERENCE, REFERENCE.with_suffix('.json'), OPERANDS, *candidate_paths.values()]},
        "source_sha256": {"analysis": sha(__file__), "exp2_probe": sha(ROOT / "experiments/bf16_attention/exp2_probe.rs")},
        "rust_exp2_binary_sha256": sha(args.exp2_binary), "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "rust_exp2_distinct_arguments": len(argument_bits), "records": records,
        "limitations": ["Saved GPU tile intermediates came from independent torch.bmm/torch.exp2. Their final selected head vectors match Flex but internal tensors are not proven equal to fused Flex.",
                        "The independent tail QK/PV used16 live keys, while compiled Flex uses64 keys with masking. Counterfactuals establish causes within the saved oracle comparison only.",
                        "GPU exp2 behavior cannot be inferred from CPU same-argument replay; fused ex2.approx.ftz requires a hardware substage capture.",
                        "No GPU numerical acceptance policy, production arithmetic, sampling scope or old evidence was changed."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(args.output)


if __name__ == '__main__': main()
