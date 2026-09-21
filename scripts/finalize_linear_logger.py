#!/usr/bin/env python3
"""Bind completed BLAS logs to the exact-output, unchanged-kernel replay."""
import json
import pathlib
import re

from fetch_reference import sha256


def main():
    folder = pathlib.Path("artifacts/reference/linear-logger-fp32")
    target = pathlib.Path("reference/linear-cublas-logged-algorithms-fp32.json")
    if target.exists():
        raise ValueError("Preserve prior logger receipts")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if not all(record["matches_original_kernel_names_grids_blocks"] and record["output_matches_frozen_bits"] for record in report["records"]):
        raise ValueError("Replay changed outputs or observed kernels")
    log = folder / f"cublaslt_{report['pid']}.log"
    traces = [(line_number, line) for line_number, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1)
              if "[Trace][cublasLtSSSMatmul]" in line]
    if len(traces) != 8:
        raise ValueError("Expected exactly eight ordered execution-linked Lt trace messages")
    cublas = (folder / "cublas.log").read_text(encoding="utf-8")
    api = "function cublasStatus_t cublasSgemm_v2("
    if cublas.count(api) != 8:
        raise ValueError("Expected eight public SGEMM calls")
    records = []
    for index, source in enumerate(report["records"]):
        case_traces = traces[index * 4:(index + 1) * 4]
        configs = []
        for line_number, line in case_traces:
            if f"[{report['pid']}]" not in line:
                raise ValueError("Wrong logger PID")
            expected_k = source["input_shape"][1]
            expected_n = source["weight_shape"][0]
            if f"Adesc=[type=R_32F rows={expected_k} cols={expected_n} ld={expected_k}]" not in line or f"Bdesc=[type=R_32F rows={expected_k} cols=144 ld={expected_k}]" not in line:
                raise ValueError("Ordered trace matrix identities differ")
            match = re.search(r"algo=\[algoId=(\d+) tile=(\S+) reductionScheme=(\S+) numSplitsK=(\d+)\].*?workSpaceSizeInBytes=(\d+)", line)
            if not match:
                raise ValueError("Execution-linked algorithm fields unavailable")
            algorithm, tile, reduction, splits, workspace = match.groups()
            configs.append({"algo_id": int(algorithm), "tile": tile, "reduction_scheme": reduction,
                            "num_splits_k": int(splits), "workspace_bytes": int(workspace)})
        if any(value != configs[0] for value in configs):
            raise ValueError("Logged execution configuration changed within one case")
        records.append({"name": source["name"], "public_api": "cublasSgemm_v2", "executed_configuration": configs[0],
                        "execution_log_lines": [number for number, _ in case_traces],
                        "ordered_calls": ["warmup1", "warmup2", "warmup3", "profiled"],
                        "output_matches_frozen_bits": True, "kernel_names_grids_blocks_unchanged": True})
    receipt = {"schema_version": 1, "status": "execution_linked_algorithm_fields_observed",
               "pid": report["pid"], "environment": report["environment"], "records": records,
               "source_fixture_sha256": report["source_fixture_sha256"],
               "original_profiler_sha256": report["original_profiler_sha256"],
               "evidence_files": {file.name: {"sha256": sha256(file), "bytes": file.stat().st_size} for file in sorted(folder.iterdir()) if file.is_file()},
               "finalizer_sha256": sha256(__file__),
               "qualification": "Fields are taken from actual execution-linked supported logs, with all warmup/profiled calls joined by PID, order and matrix dimensions. No algorithm override or grid-derived split inference. Exact K partition boundaries, within-partition FP32 accumulation order, stage/custom-option fields and reduction instruction ordering remain unobserved. Not a performance measurement."}
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "records": records}, indent=2))


if __name__ == "__main__":
    main()
