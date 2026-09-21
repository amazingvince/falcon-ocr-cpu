"""Independent, CPU-only checks of the bounded RMS output-consistency search.

The oracle uses integer significands and ties-to-even shifts, not the capture
script's midpoint division or Python float multiplication. These tests do not
observe a GPU scale or imply that an inferred scale is an actual CUDA rstd.
"""
import importlib.util
from pathlib import Path
import random
import struct
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rms_capture_under_test", HERE / "capture_scales.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits(raw):
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def rounded_shift(integer, shift):
    if shift <= 0:
        return integer << -shift
    quotient, remainder = divmod(integer, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + (remainder > half or (remainder == half and quotient & 1))


def exact_product_bits(left, right):
    """One IEEE binary32 RNE multiplication, including signed zero/subnormals."""
    sign = (left ^ right) & 0x80000000
    def decode(raw):
        exponent = (raw >> 23) & 255
        if exponent == 255:
            raise ValueError("finite operands required")
        significand = raw & 0x7fffff
        return ((significand | 0x800000, exponent - 150)
                if exponent else (significand, -149))
    a, ae = decode(left)
    b, be = decode(right)
    product, exponent = a * b, ae + be
    if not product:
        return sign
    if product.bit_length() - 1 + exponent < -126:
        return sign | rounded_shift(product, -149 - exponent)
    shift = product.bit_length() - 24
    significand = rounded_shift(product, shift)
    if significand == 1 << 24:
        significand >>= 1
        shift += 1
    encoded_exponent = exponent + shift + 150
    if encoded_exponent >= 255:
        return sign | 0x7f800000
    return sign | (encoded_exponent << 23) | (significand & 0x7fffff)


def anchor_interval(input_bits, target_bits):
    """Binary-search all positive finite F32 scales using exact products."""
    input_bits &= 0x7fffffff
    target_bits &= 0x7fffffff
    def first_above(strict):
        lo, hi = 1, 0x7f800000
        while lo < hi:
            mid = (lo + hi) // 2
            actual = exact_product_bits(input_bits, mid)
            predicate = actual > target_bits if strict else actual >= target_bits
            if predicate:
                hi = mid
            else:
                lo = mid + 1
        return lo
    return first_above(False), first_above(True) - 1


class ScaleSearchTests(unittest.TestCase):
    def assert_complete(self, inputs, expected):
        actual = capture.scale_candidates(inputs, expected)
        raw_inputs = [float_bits(x) for x in inputs]
        targets = [float_bits(x) for x in expected]
        anchor = max(range(len(inputs)), key=lambda index: abs(inputs[index]))
        lo, hi = anchor_interval(raw_inputs[anchor], targets[anchor])
        self.assertLessEqual(hi - lo, 8)
        oracle = [scale for scale in range(lo, hi + 1)
                  if all(exact_product_bits(x, scale) == y
                         for x, y in zip(raw_inputs, targets))]
        self.assertEqual(actual["consistent_scale_bits"], oracle)
        if lo <= hi:
            self.assertLessEqual(actual["candidate_bits_range"][0], lo)
            self.assertGreaterEqual(actual["candidate_bits_range"][1], hi)
        return actual

    def test_exact_oracle_known_rounding_and_signed_zero(self):
        self.assertEqual(exact_product_bits(0x3f800000, 0x3f800001), 0x3f800001)
        self.assertEqual(exact_product_bits(0x80000000, 0x3f800000), 0x80000000)
        self.assertEqual(exact_product_bits(1, 0x3f000000), 0)
        self.assertEqual(exact_product_bits(3, 0x3f000000), 2)
        self.assertEqual(exact_product_bits(0x00800000, 0x3f000000), 0x00400000)
        self.assertEqual(exact_product_bits(0x7f7fffff, 0x40000000), 0x7f800000)

    def test_normal_random_rows_and_power_of_two_boundaries(self):
        rng = random.Random(0x7681444)
        scales = [0x3effffff, 0x3f000000, 0x3f000001,
                  0x3f7fffff, 0x3f800000, 0x3f800001,
                  0x3fffffff, 0x40000000, 0x40000001]
        scales += [rng.randrange(0x3d800000, 0x41800000) for _ in range(24)]
        for scale in scales:
            with self.subTest(scale=hex(scale)):
                raw = [float_bits(8.0)] + [float_bits(rng.uniform(-4.0, 4.0)) for _ in range(767)]
                expected = [from_bits(exact_product_bits(x, scale)) for x in raw]
                result = self.assert_complete([from_bits(x) for x in raw], expected)
                self.assertIn(scale, result["consistent_scale_bits"])

    def test_ties_even_and_odd_normal_outputs(self):
        # 1.5 times adjacent floats around 1 creates exact half-ULP products.
        for scale in range(0x3f800000, 0x3f800009):
            raw = [float_bits(1.5)] * 768
            expected = [from_bits(exact_product_bits(raw[0], scale))] * 768
            self.assert_complete([1.5] * 768, expected)

    def test_subnormal_products_with_normal_anchor(self):
        raw = [float_bits(2.0), 0, 0x80000000, 1, 3, 0x80000003, 0x00800000] * 109
        raw += [float_bits(2.0)] * (768 - len(raw))
        scale = 0x3f000000
        expected = [from_bits(exact_product_bits(x, scale)) for x in raw]
        result = self.assert_complete([from_bits(x) for x in raw], expected)
        self.assertEqual(result["consistent_scale_bits"], [scale])

    def test_smallest_positive_scale_with_normal_anchor(self):
        raw = [float_bits(2.0 ** 120)] + [0] * 767
        expected = [from_bits(exact_product_bits(x, 1)) for x in raw]
        self.assertEqual(self.assert_complete([from_bits(x) for x in raw], expected)
                         ["consistent_scale_bits"], [1])

    def test_inconsistent_output_and_signed_zero_are_rejected(self):
        inputs, expected = [1.0] * 768, [1.0] * 768
        expected[13] = from_bits(0x3f800001)
        self.assertEqual(self.assert_complete(inputs, expected)["consistent_scale_bits"], [])
        inputs[13], expected[13] = -0.0, 0.0
        self.assertEqual(self.assert_complete(inputs, expected)["consistent_scale_bits"], [])
        expected[13] = -0.0
        self.assertEqual(self.assert_complete(inputs, expected)["consistent_scale_bits"], [0x3f800000])

    def test_unsupported_anchor_fails_closed(self):
        for inputs, expected in [([0.0] * 768, [0.0] * 768),
                                 ([1.0] * 768, [from_bits(1)] * 768),
                                 ([1.0] * 768, [float("inf")] * 768)]:
            with self.assertRaises(ValueError):
                capture.scale_candidates(inputs, expected)


if __name__ == "__main__":
    unittest.main()
