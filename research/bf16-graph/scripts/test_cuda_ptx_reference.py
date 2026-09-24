import ctypes as C
import pathlib
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from cuda_ptx_reference import DriverModule, elf_sections

ROOT = pathlib.Path(__file__).resolve().parents[3]


class Pointer:
    def __init__(self, value):
        self.value = value

    def data_ptr(self):
        return self.value


class AbiTests(unittest.TestCase):
    def module(self):
        module = DriverModule.__new__(DriverModule)
        module._launch = lambda *args: setattr(self, "captured", args)
        return module

    def test_attention_typed_kernel_parameters(self):
        self.module().launch_attention([Pointer(0x10000 + 0x100 * n) for n in range(14)], 7, 27136)
        values, grid, block, stream, shared = self.captured
        self.assertEqual([C.sizeof(value) for value in values], [8] * 14 + [4] * 5 + [8] * 2)
        self.assertEqual([value.value for value in values[14:]], [144, 16384, 2, 2, 2, 0, 0])
        self.assertEqual((grid, block, stream, shared), ([2, 1, 16], [128, 1, 1], 7, 27136))

    def test_natural_lse_distinct_i64_size(self):
        self.module().launch_natural_lse(Pointer(1024), Pointer(2048), 9)
        values, grid, block, stream, shared = self.captured
        self.assertEqual([C.sizeof(value) for value in values], [8, 8, 8, 4, 8, 8])
        self.assertEqual([value.value for value in values], [1024, 2048, 144, 2304, 0, 0])
        self.assertEqual((grid, block, stream, shared), ([18, 1, 1], [128, 1, 1], 9, 0))

    def test_bad_attention_pointer_count_rejected(self):
        with self.assertRaises(ValueError):
            self.module().launch_attention([Pointer(1024)] * 13, 0, 27136)

    def test_bad_elf_rejected(self):
        with self.assertRaises(ValueError):
            elf_sections(b"not ELF")

    @unittest.skipUnless((ROOT / "artifacts/reference/bf16-fused-substages-v5/original-0.cubin").is_file(), "Pinned cubin required")
    def test_real_code_section_and_truncation(self):
        data = (ROOT / "artifacts/reference/bf16-fused-substages-v5/original-0.cubin").read_bytes()
        self.assertEqual(len(elf_sections(data)[".text.triton_tem_fused_flex_attention_0"]), 202624)
        with self.assertRaises(ValueError):
            elf_sections(data[:1024])


if __name__ == "__main__":
    unittest.main()
