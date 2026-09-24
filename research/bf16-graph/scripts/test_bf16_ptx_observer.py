import ctypes
import json
import pathlib
import re
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

import instrument_bf16_ptx as instrument
import bf16_ptx_observer_support as support


ROOT = pathlib.Path(__file__).resolve().parents[3]


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (ROOT / "artifacts/reference/bf16-ptx-roundtrip-v1/original.ptx").read_bytes()
        cls.candidate, cls.mapping = instrument.build(cls.original)

    def test_restore_and_frozen_scope(self):
        self.assertEqual(instrument.restore(self.candidate), self.original)
        spec = json.loads((ROOT / "reference/bf16-fused-substage-export-spec-v3.json").read_bytes())
        old = [(int(p["case"].split(".")[-1]), p["query_row"], p["head"]) for p in spec["inputs"]["probes"]]
        self.assertEqual(instrument.PROBES, old)
        with self.assertRaises(ValueError):
            instrument.build(self.original.replace(b".target sm_89", b".target sm_80", 1))

    def test_insertions_only_write_observer_state_or_global_copy(self):
        inside = False
        allowed = {"ld.param.u64", "ld.global.u32", "mov.u32", "mov.u64", "and.b32", "mul.wide.u32",
                   "setp.eq.u32", "and.pred", "add.u64", "setp.lt.u32", "st.global.v2.b32", "add.u32"}
        for line in self.candidate.decode().splitlines():
            if line.startswith(instrument.BEGIN):
                inside = True
            elif line.startswith(instrument.END):
                inside = False
            elif inside and not line.strip().startswith(".reg"):
                m = re.match(r"\s*(?:@%obs_p\d+\s+)?(\S+)\s+([^;]+);", line)
                self.assertIsNotNone(m, line)
                self.assertIn(m[1], allowed)
                if m[1].startswith("st."):
                    self.assertRegex(m[2], r"^\[%obs_rd[23]\+\d+\], \{ %(?:r\d+|obs_r\d+), %obs_r7 \}$")
                else:
                    self.assertTrue(m[2].startswith("%obs_"), line)

    def test_mma_coordinate_bijection(self):
        seen = set()
        for row in range(128):
            mapping = instrument.row_mapping(row)
            columns = []
            for lane, tid in enumerate(mapping["threads"]):
                for slot, i in enumerate(mapping["register_indices"]):
                    group = (i//32)*2+(i%4)//2
                    reconstructed = (tid//32)*16+(tid%32)//4+8*(group%2)+64*(group//2)
                    self.assertEqual(row, reconstructed)
                    key = (tid, i)
                    self.assertNotIn(key, seen)
                    seen.add(key)
                    columns.append(mapping["columns_by_thread"][lane][slot])
            self.assertEqual(sorted(columns), list(range(64)))
        self.assertEqual(len(seen), 8192)

    def make_buffer(self, layer):
        words = np.zeros(instrument.BUFFER_BYTES//4, dtype=np.uint32)
        words[0] = layer
        cells = words.reshape(-1, 2)
        for cell in support.expected_cells(layer):
            cells[cell] = [0x3f000000, 1]
        def set_cell(offset, value):
            cells[offset//8, 0] = value
        for p, (probe_layer, row, head) in enumerate(instrument.PROBES):
            if layer != probe_layer:
                continue
            for step in range(instrument.CAPACITY):
                base = instrument.HEADER_BYTES+p*instrument.PROBE_BYTES+step*instrument.STEP_BYTES
                for field, value in [("absolute_key_start_i32", [128,192,0,64][step]),
                                     ("loop_kind_i32", step//2), ("ordinal_i32", step)]:
                    set_cell(base+instrument.FIELDS.index(field)*instrument.FIELD_BYTES, value)
                start = base+instrument.FIELDS.index("probability_bf16x2_bits")*instrument.FIELD_BYTES
                for slot in range(0,64,2):
                    set_cell(start+slot*8, 0x3f803f00)
            base = instrument.FINAL_START+p*instrument.FINAL_PROBE_BYTES
            set_cell(base+instrument.FINAL_FIELDS.index("iteration_count_i32")*instrument.FIELD_BYTES, 4)
            start = base+instrument.FINAL_FIELDS.index("raw_bf16x2_bits")*instrument.FIELD_BYTES
            for slot in range(0,64,2):
                set_cell(start+slot*8, 0x3f803f00)
        return words

    def test_complete_coverage_and_exact_bf16_unpack(self):
        for layer in [0,17,19]:
            values, report = support.decode_buffer(self.make_buffer(layer).tobytes(), layer)
            self.assertTrue(report["complete_unique_write_coverage"])
            for name, value in values.items():
                if name.endswith("bf16_promoted_f32"):
                    np.testing.assert_array_equal(value[::2], np.full(32,0.5,dtype=np.float32))
                    np.testing.assert_array_equal(value[1::2], np.ones(32,dtype=np.float32))

    def test_missing_unexpected_and_overflow_writes_rejected(self):
        for mutation in ["missing", "unexpected", "data", "overflow"]:
            words = self.make_buffer(0)
            cells = words.reshape(-1,2)
            if mutation == "missing":
                cells[min(support.expected_cells(0)),1] = 0
            elif mutation == "unexpected":
                cells[8+3*instrument.PROBE_BYTES//8,1] = 1
            elif mutation == "data":
                cells[8+3*instrument.PROBE_BYTES//8,0] = 5
            else:
                offset = instrument.FINAL_START+instrument.FINAL_FIELDS.index("iteration_count_i32")*instrument.FIELD_BYTES
                cells[offset//8,0] = 5
            with self.assertRaises(ValueError, msg=mutation):
                support.decode_buffer(words.tobytes(),0)

    def test_unchanged_abi_with_only_observer_scratch_pointer(self):
        class Fake:
            def data_ptr(self): return 123456
            def numel(self): return instrument.BUFFER_BYTES//4
            def element_size(self): return 4
        module = object.__new__(support.ObserverModule)
        captured = []
        module._launch = lambda *args: captured.append(args)
        module.launch_observer([Fake()]*14,Fake(),111,27136)
        values,grid,block,stream,shared = captured[0]
        self.assertEqual(len(values),21)
        self.assertEqual([type(x) for x in values], [ctypes.c_uint64]*14+[ctypes.c_uint32]*5+[ctypes.c_uint64]*2)
        self.assertEqual([v.value for v in values[-2:]], [123456,0])
        self.assertEqual((grid,block,stream,shared),([2,1,16],[128,1,1],111,27136))

    def test_machine_inventory_is_sensitive_to_opcode_and_immediate(self):
        sample = "/*00a0*/ FFMA R3, R5, 0.5, R7;\n/*00b0*/ MUFU.EX2 R9, R4;"
        base = support.arithmetic_inventory(sample)
        self.assertEqual(base,support.arithmetic_inventory(sample.replace("R3","R100")))
        self.assertNotEqual(base,support.arithmetic_inventory(sample.replace("0.5","0.25")))
        self.assertNotEqual(base,support.arithmetic_inventory(sample.replace("FFMA","FADD")))


if __name__ == "__main__":
    unittest.main()
