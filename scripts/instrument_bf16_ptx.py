#!/usr/bin/env python3
"""Prepare one stores-only observer for one byte-pinned emitted PTX kernel.

This module never imports CUDA/Torch, assembles, or launches. All floating-point
values are copied from original registers. Only observer integer bookkeeping,
predicates, and writes are added; deleting marked blocks restores original bytes.
"""
import argparse
import hashlib
import json
import pathlib
import re

PTX_SHA = "d7ddae62ca0170c91a5540286d931839608a43eb567100690365b4ce235523d0"
ROUNDTRIP_SHA = "461afe06cdec2d84e8d3c509be9b36edb0f9c8017677d6dd7dc191b0908e4c6d"
PROBES = [(0, 13, 1), (0, 13, 2), (0, 53, 8), (17, 54, 10),
          (17, 124, 11), (19, 41, 6), (19, 41, 5)]
VECTOR_FIELDS = ["qk_f32", "post_mod_log2_f32", "exp2_argument_f32", "exp2_f32",
                 "probability_bf16x2_bits", "accumulator_before_alpha_f32",
                 "accumulator_after_alpha_f32", "accumulator_after_fused_pv_f32"]
SCALAR_FIELDS = ["maximum_before_f32", "maximum_after_f32", "maximum_masked_f32",
                 "alpha_argument_f32", "alpha_f32", "denominator_before_f32",
                 "local_probability_sum_f32", "denominator_after_f32",
                 "absolute_key_start_i32", "loop_kind_i32", "ordinal_i32"]
FIELDS = VECTOR_FIELDS + SCALAR_FIELDS
FINAL_FIELDS = ["numerator_f32", "division_denominator_f32", "quotient_f32",
                "raw_bf16x2_bits", "raw_denominator_f32", "iteration_count_i32"]
CAPACITY = 4
HEADER_BYTES = 64
CELL_BYTES = 8  # {copied bits, validity=1}; untouched validity stays zero.
FIELD_BYTES = 64 * CELL_BYTES
STEP_BYTES = len(FIELDS) * FIELD_BYTES
PROBE_BYTES = CAPACITY * STEP_BYTES
FINAL_START = HEADER_BYTES + len(PROBES) * PROBE_BYTES
FINAL_PROBE_BYTES = len(FINAL_FIELDS) * FIELD_BYTES
BUFFER_BYTES = FINAL_START + len(PROBES) * FINAL_PROBE_BYTES
BEGIN = "// OBSERVER_BEGIN "
END = "// OBSERVER_END "


def require(test, message):
    if not test:
        raise ValueError(message)


def restore(candidate):
    """Remove only complete marked insertion blocks, preserving all other bytes."""
    lines = candidate.splitlines(keepends=True)
    out, current = [], None
    for line in lines:
        stripped = line.decode("utf-8").strip()
        if stripped.startswith(BEGIN):
            require(current is None, "Nested observer block")
            current = stripped[len(BEGIN):]
        elif stripped.startswith(END):
            require(current == stripped[len(END):], "Unmatched observer block")
            current = None
        elif current is None:
            out.append(line)
    require(current is None, "Unclosed observer block")
    return b"".join(out)


def row_mapping(row):
    group = (row // 64) * 2 + ((row % 16) // 8)
    warp = (row % 64) // 16
    first_thread = warp * 32 + (row % 8) * 4
    indices = [i for i in range(64) if (i // 32) * 2 + ((i % 4) // 2) == group]
    return {"fragment_group": group, "warp": warp, "threads": list(range(first_thread, first_thread + 4)),
            "register_indices": indices,
            "columns_by_thread": [[2 * lane + 8 * (j // 2) + j % 2 for j in range(16)] for lane in range(4)]}


def parse_ops(lines):
    result, loc = [], None
    for line_number, line in enumerate(lines):
        m = re.match(r"\s*\.loc\s+(\d+) (\d+) (\d+)", line)
        if m:
            loc = tuple(map(int, m.groups()))
        m = re.match(r"\s*([a-z][\w.]*)\s+([^;]+);", line)
        if m:
            result.append({"line": line_number, "loc": loc, "op": m[1],
                           "args": [v.strip() for v in m[2].split(",")]})
    return result


def build(original):
    require(hashlib.sha256(original).hexdigest() == PTX_SHA, "Original PTX identity changed")
    require(b"%obs_" not in original and BEGIN.encode() not in original, "Already instrumented")
    lines = original.decode("utf-8").splitlines(keepends=True)
    newline = "\r\n" if original.count(b"\r\n") else "\n"
    ops = parse_ops(lines)
    insertions, sites, mappings = {}, [], []

    def select(loc, opcode, expected):
        chosen = [op for op in ops if op["loc"] == loc and op["op"] == opcode]
        require(len(chosen) == expected, f"Unexpected {loc}/{opcode} count")
        return chosen

    def add(index, label, body):
        require(label not in [x["label"] for x in sites], "Duplicate site label")
        block = [BEGIN + label, *body, END + label]
        insertions.setdefault(index, []).append(newline.join(block) + newline)
        sites.append({"label": label, "before_original_line_1based": index + 1,
                      "original_next_line": lines[index].rstrip() if index < len(lines) else None})

    # The unused original scratch parameter supplies a valid observer allocation.
    # Its first word is the case layer id. No kernel parameter or original ABI is changed.
    require(original.count(b"triton_tem_fused_flex_attention_0_param_19") == 1, "Scratch param19 is used")
    require(original.count(b"triton_tem_fused_flex_attention_0_param_20") == 1, "Scratch param20 is used")
    first_instruction = next(i for i, line in enumerate(lines) if "ld.param.b32" in line)
    init = ["\t.reg .pred %obs_p<24>;", "\t.reg .b32 %obs_r<10>;", "\t.reg .b64 %obs_rd<4>;",
            "\tld.param.u64 %obs_rd0, [triton_tem_fused_flex_attention_0_param_19];",
            "\tld.global.u32 %obs_r3, [%obs_rd0];", "\tmov.u32 %obs_r0, %tid.x;",
            "\tmov.u32 %obs_r1, %ctaid.x;", "\tmov.u32 %obs_r2, %ctaid.z;",
            "\tand.b32 %obs_r4, %obs_r0, 3;", "\tmov.u32 %obs_r5, 0;",
            "\tmov.u32 %obs_r7, 1;", "\tand.b32 %obs_r6, %obs_r0, 124;",
            "\tmul.wide.u32 %obs_rd1, %obs_r4, 128;", "\tsetp.eq.u32 %obs_p21, %obs_r1, 0;"]
    for p, (layer, row, head) in enumerate(PROBES):
        mapping = row_mapping(row)
        mappings.append({"probe": p, "layer": layer, "row": row, "head": head, **mapping})
        init += [f"\tsetp.eq.u32 %obs_p15, %obs_r3, {layer};",
                 f"\tsetp.eq.u32 %obs_p16, %obs_r2, {head};",
                 f"\tsetp.eq.u32 %obs_p17, %obs_r6, {mapping['threads'][0]};",
                 f"\tand.pred %obs_p{p}, %obs_p15, %obs_p16;",
                 f"\tand.pred %obs_p{p}, %obs_p{p}, %obs_p17;",
                 f"\tand.pred %obs_p{p}, %obs_p{p}, %obs_p21;",
                 "\tsetp.eq.u32 %obs_p18, %obs_r4, 0;",
                 f"\tand.pred %obs_p{p+7}, %obs_p{p}, %obs_p18;"]
    add(first_instruction, "declarations_and_predicates", init)

    def capture(index, label, vectors=None, scalars=None, final=False):
        vectors, scalars = vectors or {}, scalars or {}
        body = []
        if final:
            body += ["\tmov.u64 %obs_rd2, %obs_rd0;"]
        else:
            body += ["\tmul.wide.u32 %obs_rd2, %obs_r5, " + str(STEP_BYTES) + ";",
                     "\tadd.u64 %obs_rd2, %obs_rd0, %obs_rd2;",
                     f"\tsetp.lt.u32 %obs_p19, %obs_r5, {CAPACITY};"]
        body += ["\tadd.u64 %obs_rd3, %obs_rd2, %obs_rd1;"]
        field_names = FINAL_FIELDS if final else FIELDS
        details = {"vectors": {}, "scalars": {}}
        for p, mapping in enumerate(mappings):
            body += [f"\tand.pred %obs_p20, %obs_p{p}, " + (f"%obs_p{p};" if final else "%obs_p19;"),
                     f"\tand.pred %obs_p22, %obs_p{p+7}, " + (f"%obs_p{p+7};" if final else "%obs_p19;")]
            base = FINAL_START + p * FINAL_PROBE_BYTES if final else HEADER_BYTES + p * PROBE_BYTES
            group = mapping["fragment_group"]
            for field, registers in vectors.items():
                require(len(registers) == 64, "Vector mapping must describe all 64 per-thread fragment entries")
                details["vectors"][field] = registers
                for slot, register_index in enumerate(mapping["register_indices"]):
                    register = registers[register_index]
                    if register is None:  # packed BF16 pairs stored at the even slot only.
                        continue
                    offset = base + field_names.index(field) * FIELD_BYTES + slot * CELL_BYTES
                    body.append(f"\t@%obs_p20 st.global.v2.b32 [%obs_rd3+{offset}], {{ {register}, %obs_r7 }};")
            for field, registers in scalars.items():
                require(len(registers) == 4, "Scalar mapping must describe four fragment rows")
                details["scalars"][field] = registers
                register = registers[group]
                offset = base + field_names.index(field) * FIELD_BYTES
                body.append(f"\t@%obs_p22 st.global.v2.b32 [%obs_rd2+{offset}], {{ {register}, %obs_r7 }};")
        add(index, label, body)
        sites[-1]["capture"] = details

    def regs(operations, argument=0):
        return [op["args"][argument] for op in operations]

    scales = select((1, 354, 14), "mul.f32", 128)
    args = select((1, 416, 39), "sub.f32", 128)
    probabilities = select((1, 416, 21), "ex2.approx.ftz.f32", 128)
    alpha_args = select((1, 415, 31), "sub.f32", 8)
    alpha = select((1, 415, 25), "ex2.approx.ftz.f32", 8)
    denominators = select((1, 421, 24), "fma.rn.f32", 8)
    acc_scales = select((1, 423, 16), "mul.f32", 128)
    casts = select((1, 427, 22), "cvt.rn.bf16x2.f32", 64)
    pv = [op for op in ops if op["loc"] == (1, 427, 56) and op["op"].startswith("mma.sync.")]
    require(len(pv) == 128, "Expected 64 fused MMA instructions per loop body")

    for loop, key_register in [(0, "%r1317"), (1, "%r2712")]:
        sc, ar, pr = scales[loop*64:(loop+1)*64], args[loop*64:(loop+1)*64], probabilities[loop*64:(loop+1)*64]
        aa, al = alpha_args[loop*4:(loop+1)*4], alpha[loop*4:(loop+1)*4]
        de, ac, ca = denominators[loop*4:(loop+1)*4], acc_scales[loop*64:(loop+1)*64], casts[loop*32:(loop+1)*32]
        require(regs(ar) == regs(pr, 1), "exp2 must consume original subtraction result")
        require(regs(aa) == regs(al, 1), "alpha must consume original subtraction result")
        packed = []
        for i, cast in enumerate(ca):
            require(cast["args"][1:] == [pr[2*i+1]["args"][0], pr[2*i]["args"][0]], "BF16 packing changed")
            packed += [cast["args"][0], None]
        # Match each denominator recurrence to its row via the alpha operand.
        ordered_de = [next(d for d in de if d["args"][2] == a["args"][0]) for a in al]
        require(len({d["line"] for d in ordered_de}) == 4, "Ambiguous denominator mapping")
        for i, op in enumerate(ac):
            group = (i//32)*2+(i%4)//2
            require(op["args"][0] == op["args"][1] and op["args"][2] == al[group]["args"][0], "Accumulator row mapping changed")
        max_after = ["%r" + str(3928+i) for i in range(4)]
        capture(sc[0]["line"], f"loop{loop}_qk", {"qk_f32": regs(sc, 1)},
                {"absolute_key_start_i32": [key_register]*4, "ordinal_i32": ["%obs_r5"]*4})
        # Loop kind is an observer integer label, never a floating-point value.
        add(sc[0]["line"], f"loop{loop}_kind_constant", [f"\tmov.u32 %obs_r8, {loop};"])
        capture(ar[0]["line"], f"loop{loop}_max_alpha", scalars={
            "maximum_before_f32": regs(aa, 1), "maximum_after_f32": max_after,
            "maximum_masked_f32": regs(aa, 2), "alpha_argument_f32": regs(aa),
            "alpha_f32": regs(al), "loop_kind_i32": ["%obs_r8"]*4})
        capture(pr[0]["line"], f"loop{loop}_exp2_arguments", {"post_mod_log2_f32": regs(ar, 1), "exp2_argument_f32": regs(ar)})
        capture(pr[-1]["line"]+1, f"loop{loop}_exp2_results", {"exp2_f32": regs(pr)})
        capture(de[0]["line"], f"loop{loop}_denominator_inputs", scalars={
            "denominator_before_f32": regs(ordered_de, 1), "local_probability_sum_f32": regs(ordered_de, 3)})
        capture(ac[0]["line"], f"loop{loop}_acc_before", {"accumulator_before_alpha_f32": regs(ac)},
                {"denominator_after_f32": regs(ordered_de)})
        capture(ac[-1]["line"]+1, f"loop{loop}_acc_scaled", {"accumulator_after_alpha_f32": regs(ac)})
        capture(ca[-1]["line"]+1, f"loop{loop}_probability_cast", {"probability_bf16x2_bits": packed})
        end = pv[(loop+1)*64-1]["line"]+1
        capture(end, f"loop{loop}_acc_after_fused_pv", {"accumulator_after_fused_pv_f32": regs(ac)})
        add(end, f"loop{loop}_advance_counter", ["\tadd.u32 %obs_r5, %obs_r5, 1;"])

    div = select((1, 208, 16), "div.full.f32", 64)
    out_cast = select((1, 218, 111), "cvt.rn.bf16x2.f32", 32)
    require(regs(div, 1) == regs(acc_scales[:64]), "Final numerator order changed")
    divisors = [div[i]["args"][2] for i in [0, 2, 32, 34]]
    capture(div[0]["line"], "final_inputs", {"numerator_f32": regs(div, 1)},
            {"division_denominator_f32": divisors, "raw_denominator_f32": [f"%r{3932+i}" for i in range(4)],
             "iteration_count_i32": ["%obs_r5"]*4}, final=True)
    capture(div[-1]["line"]+1, "final_quotient", {"quotient_f32": regs(div)}, final=True)
    packed = []
    for i, op in enumerate(out_cast):
        require(op["args"][1:] == [div[2*i+1]["args"][0], div[2*i]["args"][0]], "Final BF16 packing changed")
        packed += [op["args"][0], None]
    capture(out_cast[-1]["line"]+1, "final_bf16_cast", {"raw_bf16x2_bits": packed}, final=True)
    candidate = "".join("".join(insertions.get(i, [])) + line for i, line in enumerate(lines)).encode("utf-8")
    require(restore(candidate) == original, "Mechanical restoration failed")
    metadata = {"schema_version": 1, "original_ptx_sha256": PTX_SHA,
                "candidate_ptx_sha256": hashlib.sha256(candidate).hexdigest(),
                "restoration_exact": True, "probes": mappings, "sites": sites,
                "fields": FIELDS, "final_fields": FINAL_FIELDS, "capacity": CAPACITY,
                "buffer_bytes": BUFFER_BYTES, "header_bytes": HEADER_BYTES,
                "cell_bytes": CELL_BYTES, "field_bytes": FIELD_BYTES,
                "step_bytes": STEP_BYTES, "probe_bytes": PROBE_BYTES,
                "final_start": FINAL_START, "final_probe_bytes": FINAL_PROBE_BYTES,
                "gpu_execution": False, "native_intermediates_accepted": False}
    return candidate, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-ptx-roundtrip-v1/original.ptx"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous candidate directory")
    original = args.original.read_bytes()
    candidate, metadata = build(original)
    args.output.mkdir(parents=True)
    (args.output / "observer.ptx").write_bytes(candidate)
    (args.output / "restored.ptx").write_bytes(restore(candidate))
    (args.output / "mapping.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["candidate_ptx_sha256", "restoration_exact", "buffer_bytes", "gpu_execution"]}))


if __name__ == "__main__":
    main()
