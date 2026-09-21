"""Lightweight offline source checks only: no compiler, tensors or model calls."""
import hashlib
import io
import json
import re
import unittest
import zipfile

from patch import (BASELINE, BASELINE_ARCHIVE_SHA, BASELINE_BUILD_SHA, KERNELS_SHA,
                   HERE, CHANGED_SOURCE_NAMES, ADDED_SOURCE_NAMES, MODULE_START,
                   OLD_CALL, NEW_CALL, OLD_PV, NEW_PV, OLD_TEST_NAMES,
                   NEW_TEST_NAMES, TEST_NAMES, TEST_FILTER, TEST_FILTERS,
                   make_patch, make_sources, once)


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
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('Duplicate source archive member')
            cls.members = {name: z.read(name) for name in z.namelist()}
        if {name: hashlib.sha256(raw).hexdigest() for name, raw in cls.members.items()} != manifest['source_sha256']:
            raise ValueError('Source archive does not match build inventory')
        cls.original = cls.members['src/kernels.rs'].decode('utf-8')
        cls.template = (HERE / 'candidate.rs').read_text(encoding='utf-8')
        cls.tests = (HERE / 'tests.rs').read_text(encoding='utf-8')
        cls.patched, cls.parts = make_patch(cls.original, cls.template, cls.tests)

    def test_only_declared_runtime_member_and_single_dispatch_change(self):
        result = make_sources(self.members)
        self.assertEqual(result.keys(), self.members.keys())
        self.assertEqual(ADDED_SOURCE_NAMES, ())
        changed = tuple(name for name in result if result[name] != self.members[name])
        self.assertEqual(changed, CHANGED_SOURCE_NAMES)
        prefix = self.patched[:self.patched.index(MODULE_START)]
        self.assertEqual(prefix.replace(NEW_CALL, OLD_CALL), self.original)
        self.assertEqual(hashlib.sha256(self.original.encode()).hexdigest(), KERNELS_SHA)
        self.assertIn(self.parts['function'].replace('pub fn attention_compact_with_simd(',
                      'fn attention_compact_combined_for_staged_test('), self.patched)

    def test_exact_head_dot_wrapper_and_probability_order(self):
        self.assertEqual(self.parts['staged_head'].replace(NEW_PV, OLD_PV), self.parts['head'])
        self.assertIn(self.parts['dot64'], self.parts['candidate'])
        self.assertIn(self.parts['wrapper'], self.parts['candidate'])
        self.assertIn('for start in (0..visible_end).step_by(TILE)', self.parts['staged_head'])
        self.assertIn('const TILE: usize = 128;', self.parts['staged_head'])
        self.assertIn('let probability = (*logit - new_max).exp();\n                    denominator += probability;\n                    *logit = probability;', NEW_PV)
        self.assertLess(NEW_PV.index('denominator *= rescale;'), NEW_PV.index('for logit'))
        self.assertLess(NEW_PV.index('*logit = probability;'), NEW_PV.index('pv_tile('))
        self.assertNotIn('axpy64(', self.parts['staged_head'])

    def test_eight_separate_rescales_and_key_ordered_fmas(self):
        self.assertEqual(self.template.count('_mm256_mul_ps('), 8)
        self.assertEqual(self.template.count('_mm256_fmadd_ps('), 8)
        self.assertEqual(self.template.count('_mm256_storeu_ps('), 8)
        start = self.template.index('for (j, probability) in probabilities.iter().enumerate()')
        stop = self.template.index('_mm256_storeu_ps(out, y0)', start)
        loop = self.template[start:stop]
        self.assertNotIn('_mm256_storeu_ps', loop)
        self.assertNotIn('exp(', loop)
        self.assertNotIn('Vec', self.template)
        self.assertNotIn('dot_kernel', self.template)
        for i in range(8):
            self.assertIn(f'let mut y{i} = _mm256_mul_ps(', self.template)
            self.assertRegex(loop, rf'y{i} = _mm256_fmadd_ps\(factor, .*?, y{i}\);')

    def test_all_old_tests_retained_and_new_inventory_is_exact(self):
        old_names = re.findall(r'#\[test\]\s+fn\s+(\w+)\(', self.original[
            self.original.index('mod attention64_candidate {'):self.original.index('\n#[cfg(test)]\n#[allow(clippy::too_many_arguments)]\nfn attention_original_for_test(')])
        self.assertEqual(sorted(old_names), sorted(OLD_TEST_NAMES))
        new_names = re.findall(r'#\[test\]\s+fn\s+(\w+)\(', self.tests)
        self.assertEqual(sorted(new_names), sorted(NEW_TEST_NAMES))
        self.assertEqual(len(TEST_NAMES), 17)
        self.assertEqual(len(set(TEST_NAMES)), 17)
        for name in TEST_NAMES:
            self.assertEqual(sum(name.startswith(f) for f in TEST_FILTERS), 1)
            self.assertTrue(name.startswith(TEST_FILTER))
        self.assertNotIn('#[ignore', self.tests)
        self.assertNotIn('Model::', self.tests)
        self.assertNotIn('Runner::', self.tests)

    def test_changed_baseline_and_ambiguous_anchors_fail_closed(self):
        with self.assertRaises(ValueError):
            make_patch(self.original + '\n', self.template, self.tests)
        for marker in ['// GENERATED_WRAPPER', '// GENERATED_HEAD', '// GENERATED_DOT64', '    // GENERATED_TESTS']:
            with self.assertRaises(ValueError):
                make_patch(self.original, self.template + '\n' + marker, self.tests)
        with self.assertRaises(ValueError):
            make_sources({})
        for value in ['no matching insertion', 'duplicate duplicate']:
            with self.assertRaises(ValueError):
                once(value, 'duplicate', 'replacement')

    def test_validation_dispatch_and_unsupported_paths_are_unchanged(self):
        function = self.parts['function']
        self.assertIn('query_len == 1 && head_dim == 64 && selected == Simd::Avx2', function)
        self.assertLess(function.index('simd.resolved()'), function.index(OLD_CALL))
        self.assertIn('self.validate().expect("unsupported SIMD override")', self.original)
        self.assertLess(function.index('"compact attention query interval"'), function.index(OLD_CALL))
        self.assertIn('attention_gemm_compact(', function)
        self.assertIn('Simd::Scalar, Simd::Avx512', self.tests)
        self.assertIn('for rows in [0, 2, 3, 4, 17]', self.tests)


if __name__ == '__main__':
    unittest.main()
