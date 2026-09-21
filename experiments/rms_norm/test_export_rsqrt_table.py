"""Host-only mapping and exhaustive-range construction tests; no Torch execution."""
import importlib.util
import math
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("rsqrt_table_export_tested", Path(__file__).with_name("export_rsqrt_table.py"))
EXPORT = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
    SPEC.loader.exec_module(EXPORT)


def as_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class ExportDomainTests(unittest.TestCase):
    def test_every_exponent_and_boundary_reconstructs_original_argument(self):
        for biased in range(104, 255):
            base, shift = EXPORT.exponent_parameters(biased)
            for mantissa in EXPORT.BOUNDARIES:
                argument = as_f32((biased << 23) | mantissa)
                canonical = as_f32(EXPORT.TABLE_BEGIN + base + mantissa)
                self.assertTrue(1.0 <= canonical < 4.0)
                self.assertEqual(math.ldexp(canonical, -2 * shift), argument)

    def test_chunk_sizes_cover_exact_canonical_and_normal_ranges(self):
        for size in (4096, 65536, 1 << 20):
            EXPORT.validate_chunk_size(size)
            for length in (EXPORT.TABLE_LENGTH, EXPORT.MANTISSA_LENGTH):
                chunks = range(0, length, size)
                self.assertEqual(chunks[0], 0)
                self.assertEqual(chunks[-1] + size, length)
                self.assertEqual(len(chunks) * size, length)
        self.assertEqual(len(EXPORT.EXPONENTS) * EXPORT.MANTISSA_LENGTH, 1_266_679_808)
        self.assertEqual(EXPORT.TABLE_END - EXPORT.TABLE_BEGIN, 16_777_216)
        self.assertEqual(len(EXPORT.EXPONENTS) * len(EXPORT.BOUNDARIES), 1057)

    def test_invalid_sizes_and_exponents_rejected(self):
        for size in (0, 1, 4095, 4097, (1 << 20) + 1, 2.0, True):
            with self.assertRaises(ValueError):
                EXPORT.validate_chunk_size(size)
        for exponent in (0, 103, 255, 104.0, True):
            with self.assertRaises(ValueError):
                EXPORT.exponent_parameters(exponent)


if __name__ == "__main__":
    unittest.main()
