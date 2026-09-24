"""Host-only checks for the prospective, caller-owned W2 scratch observation."""
import hashlib
import json
import re
from pathlib import Path

CASE = "prefill.layer.12.w2"
M, N, K, SPLITS = 768, 144, 2304, 14
WORKSPACE_BYTES = 32 * 1024 * 1024
SELECTED_BYTES = SPLITS * N * M * 4
CANARY_BYTE = 0xA5
PINS = {
    "reference/linear-cublas-logged-algorithms-fp32.json": "d891299cf3c327285c26f6befbd3a4449f50bfbce23f45baed8de6d5b47ff646",
    "reference/linear-cuda-kernel-identities-fp32.json": "0f1c9076febb406e1f8140ea4e06774e109d7edcc19f016bf2c155cb8fb4dd07",
    "artifacts/reference/linear-logger-fp32/report.json": "6d19ce3705c2180d9cd3469198fb3bcf08db76ce4780cdbc8efb4d751b51aa86",
    "artifacts/reference/linear-operators.json": "2cd585ef48c59c9b821e0eae770b2e9c643c4697d239f4b63cac12508cf0e60f",
    "artifacts/reference/linear-operators.safetensors": "a8b9c60e8f5da012efca80fa6955edff814616f6372668229e0aba74e6f3d338",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def verify_pins(root):
    actual = {name: sha256(root / name) for name in PINS}
    require(actual == PINS, "Pinned fixture or execution evidence changed")
    return actual


def signature(kernels):
    return [(x["name"], x["arguments"].get("grid"), x["arguments"].get("block")) for x in kernels]


def parse_logs(classic, lt, calls):
    """Fail closed on public-call/configuration drift; logs must be process-finalized."""
    records = re.split(r"(?=I! cuBLAS \(v)", classic)
    gemms = [r for r in records if "function cublasStatus_t cublasSgemm_v2(" in r]
    require(len(gemms) == calls, "Unexpected SGEMM call count")
    expected = {"transa": "CUBLAS_OP_T(1)", "transb": "CUBLAS_OP_N(0)",
                "m": "768", "n": "144", "k": "2304", "lda": "2304", "ldb": "2304", "ldc": "768"}
    for record in gemms:
        for key, value in expected.items():
            fields = re.findall(r"^i!  " + key + r": type=[^;]+; val=(.*)$", record, re.M)
            require(fields == [value], "Public SGEMM argument changed: " + key)
        require("(defaultStream); MathMode=CUBLAS_DEFAULT_MATH" in record, "Stream/math changed")
    spaces = [r for r in records if "function cublasStatus_t cublasSetWorkspace_v2(" in r]
    require(len(spaces) == calls and all("workspaceSizeInBytes: type=SOME TYPE; val=33554432" in r for r in spaces),
            "Provided workspace capacity/call count changed")
    traces = [line for line in lt.splitlines() if "[Trace][cublasLtSSSMatmul]" in line]
    require(len(traces) == calls, "Unexpected executed Lt trace count")
    tokens = ["Adesc=[type=R_32F rows=2304 cols=768 ld=2304]", "Bdesc=[type=R_32F rows=2304 cols=144 ld=2304]",
              "Cdesc=[type=R_32F rows=768 cols=144 ld=768]", "Ddesc=[type=R_32F rows=768 cols=144 ld=768]",
              "computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T smCountTarget=128]",
              "algo=[algoId=0 tile=MATMUL_TILE_128x64 reductionScheme=REDUCTION_SCHEME_COMPUTE_TYPE numSplitsK=14]",
              "workSpaceSizeInBytes=6193152 beta=0 outOfPlace=0 stream=0X0"]
    for line in traces:
        require(all(token in line for token in tokens), "Executed Lt configuration differs")
    heuristics = [line for line in lt.splitlines() if "[Api][cublasLtSSSMatmulAlgoGetHeuristic]" in line]
    require(len(heuristics) == calls, "Unexpected heuristic call count")
    for line in heuristics:
        require("maxWorkspaceSizeinBytes=33554432 " in line, "Heuristic workspace capacity differs")
        require(all(f"minBytesAlignment{x}=16" in line for x in "ABCD"), "Alignment classes differ")
    return {"calls": calls, "public_arguments": expected, "provided_workspace_bytes": WORKSPACE_BYTES,
            "algo_id": 0, "tile": "128x64", "reduction": "COMPUTE_TYPE", "split_count": SPLITS,
            "selected_workspace_bytes": SELECTED_BYTES, "stream": "NULL", "math": "DEFAULT"}


def basis_operands(group):
    import numpy as np
    require(type(group) is int and 0 <= group < 3, "Invalid fixed basis group")
    x = np.broadcast_to(np.arange(1, N + 1, dtype=np.float32)[:, None], (N, K)).copy()
    w = np.zeros((M, K), dtype=np.float32)
    w[np.arange(M), group * M + np.arange(M)] = np.arange(1, M + 1, dtype=np.float32)
    expected = np.arange(1, N + 1, dtype=np.float32)[:, None] * np.arange(1, M + 1, dtype=np.float32)[None, :]
    return x, w, expected


def observe_layout(paths):
    """Test an explicit layout hypothesis. Never search offsets/permutations/layouts."""
    import numpy as np
    require(len(paths) == 3, "Need all three fixed groups for complete basis agreement")
    groups, membership, valid = [], [], True
    expected = np.arange(1, N + 1, dtype=np.float32)[:, None] * np.arange(1, M + 1, dtype=np.float32)[None, :]
    labels = expected.view(np.uint32)[None, :, :]
    for group, path in enumerate(paths):
        data = Path(path).read_bytes()
        require(len(data) == WORKSPACE_BYTES, "Owned workspace capture has wrong length")
        raw = np.frombuffer(data[:SELECTED_BYTES], dtype="<u4").reshape(SPLITS, N, M)
        matches = raw == labels
        unexpected = int(np.count_nonzero((raw != 0) & ~matches))
        wrong_count = int(np.count_nonzero(matches.sum(axis=0) != 1))
        slots = matches.argmax(axis=0)
        row_disagreement = int(np.count_nonzero(slots != slots[:1]))
        passed = not (unexpected or wrong_count or row_disagreement)
        valid &= passed
        groups.append({"group": group, "k_start": group * M, "k_end_exclusive": (group + 1) * M,
                       "unexpected_value_elements": unexpected, "wrong_nonzero_count_outputs": wrong_count,
                       "row_membership_disagreements": row_disagreement, "hypothesis_agrees": passed,
                       "captured_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        membership.extend(int(v) for v in slots[0])
    partitions = None
    if valid:
        partitions = []
        for slot in range(SPLITS):
            keys = [k for k, value in enumerate(membership) if value == slot]
            runs = []
            for k in keys:
                if runs and runs[-1][1] == k:
                    runs[-1][1] = k + 1
                else:
                    runs.append([k, k + 1])
            partitions.append({"slot": slot, "count": len(keys), "k_runs_half_open": runs,
                               "contiguous": len(runs) <= 1})
    return {"status": "hypothesis_agrees_conditionally" if valid else "layout_not_established",
            "hypothesis": "byte offset 0; little-endian FP32; [14,144,768] partial-major then output/token-row-major; exact positive zeros outside one matching partial per output",
            "groups": groups, "complete_basis_agreement": bool(valid),
            "conditional_k_membership": membership if valid else None, "conditional_partitions": partitions,
            "limitation": "Prospectively refined from unit weights to exact channel codes 1..768: every token row has distinct channel outputs. This checks the complete stated layout on all basis inputs, but is still conditional observational evidence, not an ABI guarantee. Product-code collisions across different token/channel pairs remain possible. No accumulation/reduction order is inferred."}
