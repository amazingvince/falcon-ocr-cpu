#!/usr/bin/env python3
"""Fail-closed structural audit of pinned attention denominator reductions.

This checks reduction structure, not equality of every internal instruction or
leaf value. Complete-output equality remains a separate mandatory gate.
"""
import ast
import hashlib
import json
import pathlib
import re


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _sum_source_line(module):
    values = []
    for node in ast.walk(ast.parse(pathlib.Path(module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "triton":
            source = ast.literal_eval(node.args[1])
            values.extend(index for index, line in enumerate(source.splitlines(), 1)
                          if line.strip() == "l_i = l_i * alpha + tl.sum(p, 1)")
    if len(values) != 1:
        raise ValueError("Native denominator source anchor is not unique")
    return values[0]


def _ir_sums(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    regions = re.findall(r'"tt.reduce"\([^\n]+\n.*?\}\) : [^\n]+', text, flags=re.S)
    sums = [region for region in regions if "arith.addf" in region]
    descriptors = dict(re.findall(r"^(#[\w]+) = (#ttg.nvidia_mma<[^\n]+>)$", text, flags=re.M))
    encodings = []
    for region in sums:
        found = re.search(r'\}\) : \(tensor<128x64xf32, (#[\w]+)>\) -> tensor<128xf32, #ttg.slice<\{dim = 1, parent = (#[\w]+)\}>>', region)
        if not found or found[1] != found[2] or found[1] not in descriptors:
            encodings.append("not_original_MMA_add_reduction")
        else:
            encodings.append(descriptors[found[1]])
    return encodings


def _ptx_sums(path, source_line):
    definitions, roots = {}, []
    location = ""
    serial = 0
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(".loc"):
            location = line
        match = re.match(r"\s*(\w[\w.]*)\s+(%\w+),\s*(.*);", line)
        if not match:
            continue
        opcode, destination, operands = match.groups()
        args = [value.strip() for value in operands.split(",")]
        get = lambda value: definitions.get(value, ("external", value))
        serial += 1
        if opcode == "ex2.approx.ftz.f32":
            node = ("exp2", serial)
        elif opcode == "add.f32":
            node = ("add", serial, get(args[0]), get(args[1]))
        elif opcode == "shfl.sync.bfly.b32" and len(args) == 4:
            node = ("xor", serial, args[1], args[2], args[3], get(args[0]))
        elif opcode in ["mov.b32", "mov.f32"] and len(args) == 1:
            node = get(args[0])
        else:
            node = ("other", serial, opcode)
        if opcode == "fma.rn.f32" and re.search(rf"\.py:{source_line}:24\b", location):
            roots.append(get(args[2]))
        definitions[destination] = node

    def signature(node):
        if node[0] == "exp2":
            return ["exp2"]
        if node[0] == "add":
            return ["add", signature(node[2]), signature(node[3])]
        if node[0] == "xor":
            return ["xor", *node[2:5], signature(node[5])]
        raise ValueError("Denominator reduction has an unrecognized dependency: " + str(node[:3]))

    summaries = []
    for root in roots:
        seen = {}
        def visit(node):
            if node[0] not in ["exp2", "add", "xor"]:
                raise ValueError("Non-probability leaf in denominator reduction")
            seen[node[1]] = node
            if node[0] == "add":
                visit(node[2]); visit(node[3])
            elif node[0] == "xor":
                visit(node[5])
        visit(root)
        shape = signature(root)
        summaries.append({"shape_sha256": hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest(),
                          "unique_exp2_leaves": sum(node[0] == "exp2" for node in seen.values()),
                          "unique_adds": sum(node[0] == "add" for node in seen.values()),
                          "shuffles": sorted([list(node[2:5]) for node in seen.values() if node[0] == "xor"]),
                          "shape": shape})
    return summaries


def verify_lowering(output):
    output = pathlib.Path(output)
    result = {"qualification": "Original and candidate reduction DAG shapes compare with exp2 leaves anonymized; this is a structural lowering check, not a proof of all native internal values."}
    try:
        for label in ["original", "instrumented"]:
            paths = {kind: output / (label + "-0." + kind) for kind in ["ptx", "ttgir"]}
            module = output / (label + ".py")
            result[label] = {"artifacts_sha256": {kind: _sha(path) for kind, path in paths.items()},
                             "module_sha256": _sha(module), "sum_encodings": _ir_sums(paths["ttgir"]),
                             "denominator_reductions": _ptx_sums(paths["ptx"], _sum_source_line(module))}
        expected_encoding = "#ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>"
        original, candidate = result["original"], result["instrumented"]
        expected_shuffles = [["1", "31", "-1"], ["2", "31", "-1"]]
        result["checks"] = {
            "original_two_MMA_add_reductions": original["sum_encodings"] == [expected_encoding] * 2,
            "candidate_two_matching_MMA_add_reductions": candidate["sum_encodings"] == original["sum_encodings"],
            "original_eight_native_row_reductions": len(original["denominator_reductions"]) == 8,
            "original_native_local_XOR_structure": all(row["unique_exp2_leaves"] == 16 and row["unique_adds"] == 17 and row["shuffles"] == expected_shuffles for row in original["denominator_reductions"]),
            "candidate_identical_row_reduction_shapes": candidate["denominator_reductions"] == original["denominator_reductions"],
        }
        result["equal"] = all(result["checks"].values())
    except (ValueError, OSError, IndexError) as error:
        result.update(equal=False, error=str(error))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    result = verify_lowering(args.directory)
    print(json.dumps({key: value for key, value in result.items() if key not in ["original", "instrumented"]}, indent=2))
    raise SystemExit(0 if result["equal"] else 1)
