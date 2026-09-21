"""Offline source guards only: no Rust build, tensor loading or model work."""
import hashlib
import io
import json
import re
import unittest
import zipfile

import patch


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (patch.BASELINE / 'build.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != patch.BASELINE_BUILD_SHA:
            raise ValueError('Frozen build changed')
        manifest = json.loads(raw)
        raw = (patch.BASELINE / 'source.zip').read_bytes()
        if hashlib.sha256(raw).hexdigest() != patch.BASELINE_ARCHIVE_SHA:
            raise ValueError('Frozen archive changed')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive members')
            cls.original = {name: archive.read(name) for name in names}
        if {n: hashlib.sha256(v).hexdigest() for n, v in cls.original.items()} != manifest['source_sha256']:
            raise ValueError('Archive/build inventory differs')
        cls.result = patch.make_sources(cls.original)
        cls.kernel = cls.original['src/kernels.rs'].decode('utf-8')
        cls.template = (patch.HERE / 'candidate.rs').read_text(encoding='utf-8')
        cls.tests = (patch.HERE / 'tests.rs').read_text(encoding='utf-8')

    def test_exact_source_delta_and_unchanged_default_control_paths(self):
        self.assertEqual(set(self.result) - set(self.original), set(patch.ADDED_SOURCE_NAMES))
        self.assertEqual({n for n in self.original if self.original[n] != self.result[n]},
                         set(patch.CHANGED_SOURCE_NAMES))
        kernel = self.result['src/kernels.rs'].decode('utf-8')
        self.assertTrue(kernel.startswith(self.kernel))
        config = self.result['src/config.rs'].decode('utf-8')
        self.assertIn('    #[default]\n    Expanded,\n    Compact,', config)
        self.assertEqual(config.count('    TemporalCandidate,'), 1)
        self.assertNotIn('gemv_pair_candidate', kernel)
        model = self.result['src/model.rs'].decode('utf-8')
        self.assertEqual(model.count('CacheLayout::TemporalCandidate =>'), 1)
        self.assertIn('&work.q,\n                &work.k,', model)
        self.assertIn('&work.q[range.clone()],\n                    &work.k[range.clone()],', model)
        self.assertEqual(model.count('cache.append('), 3)
        self.assertIn('cache.append(&work.k, &work.v, offset, c)?;', model)
        self.assertIn('CacheLayout::Compact => LayerCache::Compact', model)

    def test_direct_head_diff_is_only_prefix_load_and_dot_block(self):
        _, candidate, _ = patch.make_patch(self.kernel, self.template, self.tests)
        start = candidate.index('            let query = qh / n_heads;')
        end = candidate.index('\n    }\n}\n', start)
        body = candidate[start:end] + '\n'
        restored = patch.once(body, patch.SPLIT_KEY_DOT_BLOCK, patch.KEY_DOT_BLOCK)
        old_start = self.kernel.index('unsafe fn compact_head(')
        old_start = self.kernel.index('            let query = qh / n_heads;', old_start)
        old_end = self.kernel.index('\n    }\n}\n', old_start)
        self.assertEqual(restored, self.kernel[old_start:old_end] + '\n')
        head = candidate[candidate.index('unsafe fn temporal_head('):end]
        self.assertNotIn('reconstructed', head)
        self.assertNotIn('dot_kernel', head)
        self.assertNotIn('axpy_kernel', head)
        self.assertNotIn('copy_from_slice', head)
        self.assertEqual(head.count('axpy64('), 1)
        self.assertIn('const TILE: usize = 128;', head)

    def test_generic_fallback_retains_original_arithmetic_body(self):
        generic = patch.generic_adapter(self.kernel)
        body = generic[generic.index('    let dot = dot_kernel(selected);'):]
        body = patch.once(body, '            let mut reconstructed = [0.0_f32; 64];\n', '')
        body = patch.once(body,
            '                        let temporal = (key * 8 + kv_head) * 32;\n                        let spatial = (key * 16 + head) * 32;\n                        reconstructed[..32].copy_from_slice(&temporal_k[temporal..temporal+32]);\n                        reconstructed[32..].copy_from_slice(&spatial_k[spatial..spatial+32]);\n                        &reconstructed[..]',
            '                        let begin = key * query_width + head * head_dim;\n                        &prefix_k[begin..begin + head_dim]')
        original = patch.compact_function(self.kernel)
        self.assertEqual(body, original[original.index('    let dot = dot_kernel(selected);'):])
        self.assertNotIn('attention64_candidate::compact(', generic)
        self.assertNotIn('attention_gemm_compact(', generic)
        self.assertIn('assert_eq!(query_len,1', generic)
        patched = self.result['src/kernels.rs'].decode('utf-8')
        adapter = patched[len(self.kernel):patched.index('\n#[cfg(target_arch = "x86_64")]\nmod temporal64 {', len(self.kernel))]
        self.assertLess(adapter.index('let selected = simd.resolved();'), adapter.index(patch.DISPATCH))
        self.assertLess(adapter.index('"compact attention query interval"'), adapter.index(patch.DISPATCH))

    def test_split_dot_same_accumulators_and_frozen_helpers(self):
        self.assertIn('for (i, key) in [(0, temporal), (32, spatial)]', self.template)
        self.assertEqual(self.template.count('_mm256_fmadd_ps('), 4)
        self.assertEqual(self.template.count('_mm256_setzero_ps()'), 4)
        self.assertIn('_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))', self.template)
        self.assertIn('_mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc))', self.template)
        _, candidate, _ = patch.make_patch(self.kernel, self.template, self.tests)
        start = self.kernel.index('#[inline(always)]\nunsafe fn dot64(')
        end = self.kernel.index('\n#[allow(clippy::too_many_arguments)]\npub(super) unsafe fn compact(', start)
        self.assertIn(self.kernel[start:end], candidate)

    def test_preserved_storage_and_exact_disjoint_test_inventory(self):
        actual = []
        groups = [(patch.TEST_FILTERS[0], self.tests),
                  (patch.TEST_FILTERS[1], self.result['src/temporal_candidate.rs'].decode('utf-8')),
                  (patch.TEST_FILTERS[2], self.result['src/temporal_model_tests.rs'].decode('utf-8'))]
        for prefix, text in groups:
            self.assertNotIn('#[ignore', text)
            actual.extend(prefix + n for n in re.findall(r'#\[test\]\s+fn\s+(\w+)\(', text))
        self.assertEqual(sorted(actual), sorted(patch.TEST_NAMES))
        self.assertEqual(len(actual), 13)
        self.assertTrue(all(sum(f in n for f in patch.TEST_FILTERS) == 1 for n in actual))
        for name, digest in patch.REUSED_SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256(self.result['src/' + name]).hexdigest(), digest)
            self.assertEqual(self.result['src/' + name], (patch.ROOT / 'experiments/cache_layout' / name).read_bytes())
        storage = self.result['src/temporal_candidate.rs'].decode('utf-8')
        self.assertIn('kernels::attention_compact_with_simd(\n                q,\n                current_expanded_k,', storage)
        self.assertLess(storage.index('Validate all duplicate bits before mutating'),
                        storage.index('self.temporal.extend_from_slice'))

    def test_changed_sources_and_ambiguous_anchors_reject(self):
        for name in patch.CHANGED_SOURCE_NAMES:
            changed = dict(self.original)
            changed[name] += b'\n'
            with self.assertRaises(ValueError):
                patch.make_sources(changed)
        changed = dict(self.original)
        changed[patch.ADDED_SOURCE_NAMES[0]] = b'collision'
        with self.assertRaises(ValueError):
            patch.make_sources(changed)
        for text in ['absent', 'anchor anchor']:
            with self.assertRaises(ValueError):
                patch.once(text, 'anchor', 'replacement')
        with self.assertRaises(ValueError):
            patch.make_patch(self.kernel, self.template + '\n// GENERATED_FIXED_HELPERS', self.tests)


if __name__ == '__main__':
    unittest.main()
