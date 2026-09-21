"""Read tensor headers and calculate payloads; no tensor quantization or inference."""
import datetime
import hashlib
import json
import pathlib
import struct

root = pathlib.Path(__file__).resolve().parents[2]
checkpoint = root / "artifacts/model/model.safetensors"
with checkpoint.open("rb") as stream:
    header_length = struct.unpack("<Q", stream.read(8))[0]
    raw_header = stream.read(header_length)
header = json.loads(raw_header)
tensors = {name: item for name, item in header.items() if name != "__metadata__"}
assert all(item["dtype"] == "F32" for item in tensors.values())
def elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n
fp32_payload = sum(elements(item["shape"]) * 4 for item in tensors.values())
groups = {}
for name, item in tensors.items():
    shape = item["shape"]
    if len(shape) != 2:
        continue
    key = name.split(".", 2)[2] if name.startswith("layers.") else name
    group = groups.setdefault(key, {"shape_out_in": shape, "matrices": 0, "weights_per_matrix": elements(shape)})
    assert group["shape_out_in"] == shape
    group["matrices"] += 1
formats = {
    "custom_q4_g32_f32scale": (32, 4, 4),
    "custom_q4_g64_f32scale": (64, 4, 4),
    "custom_q4_g128_f32scale": (128, 4, 4),
    "ggml_q4_0": (32, 4, 2),
    "ggml_q4_K": (256, 4, 16),
    "ggml_q8_0": (32, 8, 2),
    "custom_q8_g64_f32scale": (64, 8, 4),
}
for group in groups.values():
    n, k = group["shape_out_in"]
    group["fp32_payload_per_matrix"] = n * k * 4
    group["bf16_payload_per_matrix"] = n * k * 2
    group["formats"] = {}
    for name, (g, bits, meta) in formats.items():
        assert k % g == 0, (name, k)
        codes = n * k * bits // 8
        metadata = n * (k // g) * meta
        group["formats"][name] = {"code_bytes": codes, "scale_and_other_metadata_bytes": metadata,
                                   "payload_per_matrix": codes + metadata,
                                   "effective_bits_per_weight": 8 * (codes + metadata) / (n * k)}
    group["w8_per_channel_f32scale_bytes"] = n * k + n * 4
policies = {}
for policy in ["transformer_linears_only", "transformer_linears_and_output", "all_2d_weights"]:
    selected = [value for key, value in groups.items() if policy == "all_2d_weights"
                or key.startswith(("attention.", "feed_forward."))
                or (policy == "transformer_linears_and_output" and key == "output.weight")]
    selected_fp32 = sum(g["matrices"] * g["fp32_payload_per_matrix"] for g in selected)
    retained = fp32_payload - selected_fp32
    policies[policy] = {"selected_fp32_bytes": selected_fp32, "retained_fp32_bytes": retained,
                       "formats": {}}
    for name in formats:
        encoded = sum(g["matrices"] * g["formats"][name]["payload_per_matrix"] for g in selected)
        policies[policy]["formats"][name] = {"selected_quantized_bytes": encoded,
            "whole_model_tensor_payload_bytes": encoded + retained,
            "whole_model_tensor_payload_mib": (encoded + retained) / 2**20,
            "ratio_vs_all_fp32_payload": fp32_payload / (encoded + retained)}
def sha(path):
    return hashlib.sha256((root / path).read_bytes()).hexdigest()
report = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "qualification": "Payload arithmetic from actual tensor headers; no quantized checkpoint, speed, RSS, or accuracy claim.",
          "checkpoint": "artifacts/model/model.safetensors",
          "checkpoint_expected_sha256": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16",
          "verification_scope": "The earlier provenance audit hashed the entire checkpoint; this script reads only its safetensors header.",
          "tensor_header_sha256": hashlib.sha256(raw_header).hexdigest(),
          "checkpoint_file_bytes": checkpoint.stat().st_size, "safetensors_header_bytes_including_length": header_length + 8,
          "fp32_tensor_payload_bytes": fp32_payload, "non_2d_retained_fp32_bytes": fp32_payload - sum(g["matrices"] * g["fp32_payload_per_matrix"] for g in groups.values()),
          "matrix_groups": groups, "policies": policies,
          "excluded_costs": ["serialization metadata", "alignment", "allocator metadata", "ISA packing", "scratch", "KV cache", "coexisting original mapped/resident weights"],
          "isa_native_windows": json.loads((root / "artifacts/quantization/isa-native-windows.json").read_text(encoding="utf-8-sig")),
          "source_sha256": {p: sha(p) for p in ["experiments/quantization/q4_reference.rs", "experiments/quantization/isa_probe.rs", "experiments/quantization/describe_layout.py"]},
          "scalar_tests": {"command": "rustc --edition 2024 --test experiments/quantization/q4_reference.rs -C opt-level=1 -o artifacts/quantization/q4-reference-tests.exe; artifacts/quantization/q4-reference-tests.exe --test-threads 1",
                           "observed_passed": 4, "observed_failed": 0, "rustc": "1.92.0 (ded5c06cf 2025-12-08)", "host": "x86_64-pc-windows-msvc"}}
(root / "reference/quantization-feasibility-layouts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"fp32_tensor_payload_bytes": fp32_payload, "matrix_groups": groups, "policies": policies}, indent=2))
