"""Host decoding, write-coverage checks and unchanged-ABI launch for PTX observer."""
import collections
import ctypes as C
import re

import numpy as np

from cuda_ptx_reference import DriverModule
from instrument_bf16_ptx_v2 import (PROBES, FIELDS, FINAL_FIELDS, VECTOR_FIELDS, CAPACITY,
                                 HEADER_BYTES, CELL_BYTES, FIELD_BYTES, STEP_BYTES,
                                 PROBE_BYTES, FINAL_START, FINAL_PROBE_BYTES, BUFFER_BYTES,
                                 row_mapping, require, unavailable_columns)


class ObserverModule(DriverModule):
    def launch_observer(self, pointers, observer, stream, shared_bytes):
        require(len(pointers) == 14, "Observer retains fourteen tensor pointers")
        require(observer.numel() * observer.element_size() == BUFFER_BYTES, "Observer allocation size changed")
        values = [C.c_uint64(value.data_ptr()) for value in pointers]
        values += [C.c_uint32(value) for value in [144, 16384, 2, 2, 2]]
        values += [C.c_uint64(observer.data_ptr()), C.c_uint64(0)]
        self._launch(values, [2, 1, 16], [128, 1, 1], stream, shared_bytes)


def expected_cells(layer):
    expected = set()
    for p, (probe_layer, row, head) in enumerate(PROBES):
        if layer != probe_layer:
            continue
        for step in range(CAPACITY):
            for f, field in enumerate(FIELDS):
                slots = range(0, 64, 2) if field.endswith("bf16x2_bits") else range(64) if field in VECTOR_FIELDS else [0]
                base = HEADER_BYTES + p * PROBE_BYTES + step * STEP_BYTES + f * FIELD_BYTES
                for slot in slots:
                    lane, lane_slot = divmod(slot, 16)
                    column = row_mapping(row)["columns_by_thread"][lane][lane_slot]
                    if column in unavailable_columns(p, step, field):
                        continue
                    cell = (base + slot * CELL_BYTES) // CELL_BYTES
                    require(cell not in expected, "Duplicate write cell")
                    expected.add(cell)
        for f, field in enumerate(FINAL_FIELDS):
            slots = range(0, 64, 2) if field.endswith("bf16x2_bits") else [0] if field in ["division_denominator_f32", "raw_denominator_f32", "iteration_count_i32"] else range(64)
            base = FINAL_START + p * FINAL_PROBE_BYTES + f * FIELD_BYTES
            for slot in slots:
                cell = (base + slot * CELL_BYTES) // CELL_BYTES
                require(cell not in expected, "Duplicate final write cell")
                expected.add(cell)
    return expected


def decode_buffer(raw, layer):
    require(len(raw) == BUFFER_BYTES, "Observer buffer byte count changed")
    words = np.frombuffer(raw, dtype="<u4")
    require(int(words[0]) == layer and not np.any(words[1:HEADER_BYTES//4]), "Observer header changed")
    cells = words.reshape(-1, 2)
    expected = expected_cells(layer)
    actual = set(np.flatnonzero(cells[:, 1]).tolist())
    require(actual == expected, "Missing or unexpected observer writes")
    require(all(cells[i, 1] == 1 for i in actual), "Observer validity marker differs")
    untouched = np.ones(len(cells), dtype=bool)
    untouched[list(expected)] = False
    untouched[:HEADER_BYTES//CELL_BYTES] = False
    require(not np.any(cells[untouched]), "Unexpected data outside declared write cells")
    result = {}

    def read(base, field, scalar):
        data = cells[base//CELL_BYTES:base//CELL_BYTES+64, 0]
        if scalar:
            return data[:1].copy()
        columns = row_mapping(0)["columns_by_thread"]
        values = np.empty(64, dtype=np.uint32)
        if field.endswith("bf16x2_bits"):
            for lane in range(4):
                for slot in range(0, 16, 2):
                    packed = int(data[lane*16+slot])
                    values[columns[lane][slot]] = (packed & 0xffff) << 16
                    values[columns[lane][slot+1]] = packed & 0xffff0000
        else:
            for lane in range(4):
                values[columns[lane]] = data[lane*16:(lane+1)*16]
        return values

    for p, (probe_layer, row, head) in enumerate(PROBES):
        if layer != probe_layer:
            continue
        name = f"layer{layer}.row{row}.head{head}"
        for step in range(CAPACITY):
            for f, field in enumerate(FIELDS):
                bits = read(HEADER_BYTES+p*PROBE_BYTES+step*STEP_BYTES+f*FIELD_BYTES, field, field not in VECTOR_FIELDS)
                value = bits.view(np.int32 if field.endswith("_i32") else np.float32)
                output_field = field.replace("bf16x2_bits", "bf16_promoted_f32")
                if field == "exp2_argument_f32":
                    available = np.ones(64, dtype=bool)
                    missing = list(unavailable_columns(p, step, field))
                    available[missing] = False
                    # These NaNs are host placeholders, explicitly not observed
                    # native values. No unavailable argument is reconstructed.
                    bits[missing] = np.uint32(0x7fc00000)
                    result[f"{name}.tile{step}.{output_field}.available"] = available
                result[f"{name}.tile{step}.{output_field}"] = value.copy()
            require(int(result[f"{name}.tile{step}.absolute_key_start_i32"][0]) == [128,192,0,64][step], "Native absolute tile schedule changed")
            require(int(result[f"{name}.tile{step}.loop_kind_i32"][0]) == [0,0,1,1][step], "Native partial/full schedule changed")
            require(int(result[f"{name}.tile{step}.ordinal_i32"][0]) == step, "Observer ordinal changed")
        for f, field in enumerate(FINAL_FIELDS):
            scalar = field in ["division_denominator_f32", "raw_denominator_f32", "iteration_count_i32"]
            bits = read(FINAL_START+p*FINAL_PROBE_BYTES+f*FIELD_BYTES, field, scalar)
            output_field = field.replace("bf16x2_bits", "bf16_promoted_f32")
            result[f"{name}.final.{output_field}"] = bits.view(np.int32 if field.endswith("_i32") else np.float32).copy()
        require(int(result[f"{name}.final.iteration_count_i32"][0]) == CAPACITY, "Observed loop count exceeds or misses bounded capacity")
    unavailable = sum(len(unavailable_columns(p, step, "exp2_argument_f32"))
                      for p, (probe_layer, _, _) in enumerate(PROBES) if probe_layer == layer
                      for step in range(CAPACITY))
    return result, {"expected_cells": len(expected), "actual_cells": len(actual),
                    "declared_unavailable_argument_cells": unavailable,
                    "unavailable_cells_untouched": True, "complete_unique_write_coverage": True}


def arithmetic_inventory(sass):
    """Necessary instruction inventory gate, not a proof of whole-SASS equivalence.

    Original PTX restoration proves unchanged source arithmetic/dependencies.
    This complementary gate rejects changed machine FP/MMA/SFU opcode+immediate
    multiplicities; it intentionally ignores register assignment and scheduling.
    """
    counts = collections.Counter()
    for line in sass.splitlines():
        match = re.search(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s*([^;]*);", line)
        if not match:
            continue
        opcode, operands = match.groups()
        if not (opcode.startswith(("F", "H", "MUFU", "D")) and opcode not in ["DEPBAR"]):
            continue
        immediates = re.findall(r"(?<![\w])(?:-?0x[0-9a-f]+|-?\d+(?:\.\d*)?(?:e[+-]?\d+)?)(?![\w])", re.sub(r"\b(?:R|P|UR|UP)\d+\b", "REG", operands))
        counts[(opcode, tuple(immediates))] += 1
    require(counts, "No machine arithmetic inventory found")
    return [{"opcode": op, "immediates": list(imm), "count": count} for (op, imm), count in sorted(counts.items())]
