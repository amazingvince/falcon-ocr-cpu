"""Focused source guards; run only after the parent releases the quiet window."""
import hashlib
import json
import re
import unittest
import zipfile
from pathlib import Path

from patch import (BASELINE, BASELINE_BUILD_SHA, BASELINE_ARCHIVE_SHA, KERNELS_SHA,
                   INPUT_PINS, TEST_NAMES, DISPATCH, MODULE_START, make_patch, once)

HERE = Path(__file__).resolve().parent


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (BASELINE / 'build.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != BASELINE_BUILD_SHA:
            raise ValueError('Frozen combined-attention build changed')
        manifest = json.loads(raw)
        archive = (BASELINE / 'source.zip').read_bytes()
        if hashlib.sha256(archive).hexdigest() != BASELINE_ARCHIVE_SHA:
            raise ValueError('Frozen combined-attention archive changed')
        import io
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive members')
            inventory = {name: hashlib.sha256(z.read(name)).hexdigest() for name in names}
            if inventory != manifest['source_sha256']:
                raise ValueError('Archive and build source inventory differ')
            cls.original = z.read('src/kernels.rs').decode('utf-8')
        cls.template = (HERE / 'candidate.rs').read_text(encoding='utf-8')
        cls.tests = (HERE / 'tests.rs').read_text(encoding='utf-8')

    def test_only_dispatch_and_appended_candidate_oracle_change(self):
        patched, candidate, function = make_patch(self.original, self.template, self.tests)
        prefix = patched[:patched.index(MODULE_START)]
        self.assertEqual(prefix.replace(DISPATCH, ''), self.original)
        self.assertIn(function.replace('pub fn linear_with_simd(', 'fn linear_original_for_gemv_pair_test('), patched)
        self.assertIn(self.tests, candidate)
        self.assertEqual(hashlib.sha256(self.original.encode()).hexdigest(), KERNELS_SHA)

    def test_changed_baseline_and_ambiguous_insertion_reject(self):
        with self.assertRaises(ValueError):
            make_patch(self.original + '\n', self.template, self.tests)
        with self.assertRaises(ValueError):
            make_patch(self.original, self.template + '\n    // GENERATED_TESTS', self.tests)
        for text in ['no matching insertion', 'duplicate duplicate']:
            with self.assertRaises(ValueError):
                once(text, 'duplicate', 'replacement')

    def test_dispatch_scope_and_independent_accumulator_contract(self):
        self.assertIn('rows == 1 && selected == Simd::Avx2', DISPATCH)
        self.assertIn('supported_shape(in_dim, out_dim)', DISPATCH)
        self.assertIn('out.par_chunks_mut(32)', self.template)
        self.assertNotIn('dot_kernel', self.template)
        for k, n in [(768, 2048), (1024, 768), (768, 4608), (2304, 768), (768, 65536)]:
            self.assertIn(f'({k}, {n})', self.template)
        self.assertEqual(self.template.count('_mm256_fmadd_ps('), 8)
        self.assertIn('(finish(a0, a1, a2, a3), finish(b0, b1, b2, b3))', self.template)
        self.assertIn('_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))', self.template)
        self.assertNotIn('Vec', self.template)

    def test_exact_operator_inventory_and_real_operand_pins(self):
        names = re.findall(r'#\[test\]\s+fn\s+(\w+)\(', self.tests)
        self.assertEqual(sorted(names), sorted(TEST_NAMES))
        self.assertEqual(len(names), 7)
        self.assertNotIn('#[ignore', self.tests)
        self.assertNotIn('Model::load', self.tests)
        self.assertNotIn('Runner::', self.tests)
        for path, digest in INPUT_PINS.items():
            self.assertIn(path, self.tests)
            self.assertIn(digest, self.tests)
        self.assertIn('decode.1.layer.21.hidden', self.tests)
        self.assertIn('&input[112 * k..113 * k]', self.tests)


if __name__ == '__main__':
    unittest.main()
