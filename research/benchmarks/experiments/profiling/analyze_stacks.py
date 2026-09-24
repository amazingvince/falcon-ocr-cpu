#!/usr/bin/env python3
"""Summarize a PerfView CPU-stack XML export; this does not accept an ETL capture."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

PROCESS_ROOT = re.compile(r"^Process(?:32|64)?\s+.+? \((\d+)\)(?: Args:.*)?$", re.IGNORECASE)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(names):
    """Caller context only: categories deliberately do not invent phase markers."""
    joined = "\n".join(names).lower()
    if "attention_gemm" in joined:
        return "attention_gemm_call_path"
    if "attention_with_simd" in joined or "attention_compact_with_simd" in joined:
        return "attention_call_path"
    if "linear_with_simd" in joined:
        return "linear_call_path"
    if "gemm" in joined:
        return "gemm_call_path"
    if "rms_norm" in joined or "sum_squares_pairwise" in joined:
        return "normalization_call_path"
    if "squared_relu_gate" in joined:
        return "gate_call_path"
    if "dot_avx" in joined:
        return "dot_without_resolved_operator_caller"
    if "axpy_avx" in joined:
        return "axpy_without_resolved_operator_caller"
    if "rayon" in joined:
        return "rayon_without_resolved_operator_caller"
    return "other_or_unresolved"


def summarize(path, expected_pid):
    frames, stacks, ancestry = {}, {}, {}
    exclusive, inclusive, categories, roots = Counter(), Counter(), Counter(), Counter()
    time_bins = defaultdict(Counter)
    samples = 0
    metric = 0.0
    first, last = math.inf, -math.inf
    missing_stack_metric = 0.0
    project_unresolved_leaf_metric = 0.0
    project_resolved_leaf_metric = 0.0

    def chain(stack_id):
        if stack_id in ancestry:
            return ancestry[stack_id]
        result, seen = [], set()
        while stack_id >= 0:
            require(stack_id not in seen, "Cyclic exported stack")
            seen.add(stack_id)
            require(stack_id in stacks, "Missing exported stack")
            frame, stack_id = stacks[stack_id]
            require(frame in frames, "Missing exported frame")
            result.append(frames[frame])
        return tuple(result)

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        require(len(members) == 1, "Expected exactly one XML stack member")
        with archive.open(members[0]) as source:
            for _, element in ET.iterparse(source, events=("end",)):
                if element.tag == "Frame":
                    key = int(element.attrib["ID"])
                    require(key not in frames, "Duplicate frame ID")
                    frames[key] = element.text or ""
                elif element.tag == "Stack":
                    key = int(element.attrib["ID"])
                    require(key not in stacks, "Duplicate stack ID")
                    stacks[key] = (int(element.attrib["FrameID"]), int(element.attrib["CallerID"]))
                elif element.tag == "Sample":
                    weight = float(element.attrib.get("Metric", "1"))
                    time = float(element.attrib["Time"])
                    require(math.isfinite(weight) and weight > 0 and math.isfinite(time), "Invalid sample")
                    key = int(element.attrib["StackID"])
                    names = ancestry.setdefault(key, chain(key))
                    process_roots = [name for name in names if PROCESS_ROOT.fullmatch(name)]
                    require(len(process_roots) == 1 and int(PROCESS_ROOT.fullmatch(process_roots[0]).group(1)) == expected_pid,
                            "Sample lacks one unambiguous captured-PID process root")
                    samples += 1
                    metric += weight
                    first, last = min(first, time), max(last, time)
                    group = category(names)
                    categories[group] += weight
                    time_bins[int(time // 1000)][group] += weight
                    if not names:
                        missing_stack_metric += weight
                    else:
                        exclusive[names[0]] += weight
                        for name in set(names):
                            inclusive[name] += weight
                        for name in names:
                            if PROCESS_ROOT.fullmatch(name):
                                roots[name] += weight
                        leaf = names[0]
                        if "ocr_bench!" in leaf.lower():
                            if "!?" in leaf or re.search(r"!(?:0x)?[0-9a-fA-F]+$", leaf):
                                project_unresolved_leaf_metric += weight
                            else:
                                project_resolved_leaf_metric += weight
                element.clear()
    require(samples > 0 and metric > 0, "No exported CPU samples")
    require(roots and all(int(PROCESS_ROOT.fullmatch(root).group(1)) == expected_pid for root in roots),
            "Exported process roots are missing or differ from captured PID: " + repr(dict(roots)))

    def ranked(counter):
        return [{"name": name, "sampled_cpu_metric": value, "percent_of_export_metric": 100 * value / metric}
                for name, value in counter.most_common()]

    return {
        "kind": "perfview-cpu-stack-summary-v1",
        "status": "export_parsed_requires_capture_audit",
        "input": str(path.resolve()), "input_sha256": sha(path), "expected_pid": expected_pid,
        "xml_member": members[0], "frame_count": len(frames), "stack_count": len(stacks),
        "sample_count": samples, "sampled_cpu_metric_total": metric,
        "first_sample_relative_ms": first, "last_sample_relative_ms": last,
        "process_roots": dict(roots), "empty_stack_metric": missing_stack_metric,
        "project_resolved_leaf_metric": project_resolved_leaf_metric,
        "project_unresolved_leaf_metric": project_unresolved_leaf_metric,
        "exclusive_by_name": ranked(exclusive), "inclusive_by_name": ranked(inclusive),
        "exclusive_call_path_categories": ranked(categories),
        "one_second_bins": [{"trace_second": key, "categories": dict(values)} for key, values in sorted(time_bins.items())],
        "limits": [
            "CPU sample metrics are not page wall time or hardware bandwidth counters.",
            "Whole-process export includes loading, image work, prefill and decode; no exact phase markers are inferred.",
            "Caller categories are best-effort symbol-name attribution; unresolved/inlined callers remain explicit.",
            "Inclusive names overlap and must not be added as exclusive percentages.",
            "Independent exact-PID ETL loss/lifetime/stack audit and model output comparison are required.",
            "Profiled latency cannot replace the quiet unprofiled baseline."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be fresh")
    analyzer_before = sha(Path(__file__))
    before = sha(args.input)
    report = summarize(args.input, args.pid)
    require(sha(args.input) == before == report["input_sha256"], "Export changed during analysis")
    require(sha(Path(__file__)) == analyzer_before, "Analyzer changed during execution")
    report["analyzer_sha256"] = analyzer_before
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({"status": report["status"], "samples": report["sample_count"], "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
