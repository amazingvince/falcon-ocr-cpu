"""Bounded offline guards: saved source bytes only; no build/tensors/model."""
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
        base = patch.ROOT / 'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
        raw = (base / 'build.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != patch.BASELINE_BUILD_SHA:
            raise ValueError('Frozen combined build changed')
        build = json.loads(raw)
        raw = (base / 'source.zip').read_bytes()
        if hashlib.sha256(raw).hexdigest() != patch.BASELINE_ARCHIVE_SHA:
            raise ValueError('Frozen combined source archive changed')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate source archive members')
            cls.original = {n: archive.read(n) for n in names}
        if {n: hashlib.sha256(v).hexdigest() for n, v in cls.original.items()} != build['source_sha256']:
            raise ValueError('Source archive/build inventory differs')
        cls.result = patch.make_sources(cls.original)
        cls.kernel = cls.original['src/kernels.rs'].decode('utf-8')
        cls.template = (patch.HERE / 'candidate.rs').read_text(encoding='utf-8')

    def test_exact_four_changed_two_added_and_original_controls_preserved(self):
        self.assertEqual(set(self.result) - set(self.original),
                         {'src/head_contiguous_prefix.rs', 'src/head_contiguous_prefix_model_tests.rs'})
        self.assertEqual({n for n in self.original if self.original[n] != self.result[n]},
                         {'src/config.rs', 'src/lib.rs', 'src/model.rs', 'src/kernels.rs'})
        self.assertTrue(self.result['src/kernels.rs'].startswith(self.original['src/kernels.rs']))
        config = self.result['src/config.rs'].decode('utf-8')
        self.assertIn('    #[default]\n    Expanded,\n    Compact,', config)
        self.assertEqual(config.count('    HeadContiguousPrefix,'), 1)
        self.assertEqual(self.result['src/lib.rs'], self.original['src/lib.rs'] + b'\nmod head_contiguous_prefix;\n')
        original_model = self.original['src/model.rs'].decode('utf-8')
        model = self.result['src/model.rs'].decode('utf-8')
        for first, last in [
            ('            Self::Expanded { k: keys, v: values } => {', '\n        }\n    }\n\n    #[allow'),
            ('            Self::Expanded { k, v } => kernels::attention_with_simd(', '\n        }\n    }\n}\n'),
            ('                CacheLayout::Expanded => LayerCache::Expanded {', '\n            });')]:
            begin = original_model.index(first)
            body = original_model[begin:original_model.index(last, begin)]
            self.assertIn(body, model, 'Original Expanded/Compact branch arithmetic changed')
        self.assertNotIn('gemv_pair', self.result['src/kernels.rs'].decode('utf-8'))
        self.assertNotIn('TemporalCandidate', config)

    def test_direct_head_changes_only_two_address_expressions(self):
        _, candidate, _ = patch.make_patch(self.kernel, self.template)
        begin = candidate.index('            let query = qh / n_heads;')
        end = candidate.index('\n    }\n}\n', begin)
        body = candidate[begin:end] + '\n'
        restored = patch.once(body, patch.NEW_K_ADDRESS, patch.OLD_K_ADDRESS)
        restored = patch.once(restored, patch.NEW_V_ADDRESS.format(axpy='axpy64'),
                             patch.OLD_V_ADDRESS.format(axpy='axpy64'))
        old_start = self.kernel.index('unsafe fn compact_head(')
        old_start = self.kernel.index('            let query = qh / n_heads;', old_start)
        old_end = self.kernel.index('\n    }\n}\n', old_start)
        self.assertEqual(restored, self.kernel[old_start:old_end] + '\n')
        self.assertEqual(body.count('for start in (0..visible_end).step_by(TILE)'), 1)
        self.assertIn('const TILE: usize = 128;', body)
        self.assertEqual(body.count('let mut denominator = 0.0_f32;'), 1)
        self.assertEqual(body.count('let mut running_max = f32::NEG_INFINITY;'), 1)
        self.assertNotIn('copy_from_slice', body)
        self.assertNotIn('reconstructed', body)
        helper_start = self.kernel.index('#[inline(always)]\nunsafe fn dot64(')
        helper_end = self.kernel.index('\n#[allow(clippy::too_many_arguments)]\npub(super) unsafe fn compact(', helper_start)
        self.assertIn(self.kernel[helper_start:helper_end], candidate)

    def test_scalar_avx512_body_preserves_global_tiles_and_math(self):
        generic = patch.generic_adapter(self.kernel)
        body = generic[generic.index('    let dot = dot_kernel(selected);'):]
        body = patch.once(body, patch.NEW_K_ADDRESS, patch.OLD_K_ADDRESS)
        body = patch.once(body, patch.NEW_V_ADDRESS.format(axpy='axpy'),
                          patch.OLD_V_ADDRESS.format(axpy='axpy'))
        old = patch.compact_function(self.kernel)
        self.assertEqual(body, old[old.index('    let dot = dot_kernel(selected);'):])
        self.assertNotIn('attention_gemm_compact(', generic)
        self.assertNotIn('attention64_candidate::compact(', generic)
        patched = self.result['src/kernels.rs'].decode('utf-8')[len(self.kernel):]
        adapter = patched[:patched.index('\n#[cfg(target_arch = "x86_64")]\nmod head_prefix64 {')]
        self.assertLess(adapter.index('let selected = simd.resolved();'), adapter.index(patch.DISPATCH))
        self.assertLess(adapter.index('"compact attention query interval"'), adapter.index(patch.DISPATCH))
        self.assertLess(adapter.index('"head-contiguous generated V shape"'), adapter.index(patch.DISPATCH))
        self.assertIn('assert_eq!(query_len,1', adapter)
        self.assertIn('assert_eq!((n_heads,n_kv_heads,head_dim),(16,8,64)', adapter)

    def test_storage_validates_before_writes_and_has_four_buffers_only(self):
        storage = (patch.HERE / 'head_contiguous_prefix.rs').read_text(encoding='utf-8')
        fields = storage[storage.index('pub(crate) struct HeadContiguousPrefixCache'):storage.index('\nfn duplicate_heads_equal')]
        self.assertEqual(re.findall(r'(\w+): Vec<f32>', fields), ['prefix_k', 'prefix_v', 'generated_k', 'generated_v'])
        append = storage[storage.index('    pub(crate) fn append('):storage.index('\n    #[allow(clippy::too_many_arguments)]')]
        first_write = append.index('self.prefix_k.extend_from_slice')
        self.assertLess(append.index('ensure!(duplicate_heads_equal(v)'), first_write)
        self.assertLess(append.index('ensure!(prefill || duplicate_heads_equal(k)'), first_write)
        self.assertNotIn('reserve', append)
        self.assertIn('a.to_bits() == b.to_bits()', storage)
        pack = storage[storage.index('pub(crate) fn pack_prefill_values'):storage.index('\nimpl HeadContiguousPrefixCache')]
        self.assertLess(pack.index('ensure!(duplicate_heads_equal(expanded)'), pack.index('scratch.clear()'))
        self.assertLess(pack.index('ensure!(scratch.capacity()'), pack.index('scratch.clear()'))
        self.assertNotIn('.reserve', pack)
        self.assertIn('kernels::attention_compact_with_simd(q, current_expanded_k, &[], prefill_compact_v,', storage)

    def test_prefill_scratch_is_local_before_layers_and_decode_passes_empty(self):
        model = self.result['src/model.rs'].decode('utf-8')
        self.assertEqual(model.count('let mut prefill_compact_v = Vec::new();'), 1)
        self.assertEqual(model.count('prefill_compact_v.try_reserve_exact(count)'), 1)
        self.assertEqual(model.count('pack_prefill_values(&work.v, &mut prefill_compact_v)?;'), 1)
        start = model.index('let head_contiguous_prefill = offset == 0')
        reserve = model.index('prefill_compact_v.try_reserve_exact(count)', start)
        layers = model.index('for (i, layer) in self.layers.iter().enumerate()', start)
        self.assertLess(reserve, layers)
        self.assertIn('&work.q,\n                &work.k,\n                &prefill_compact_v,', model)
        self.assertIn('&work.q[range.clone()],\n                    &work.k[range.clone()],\n                    &[],', model)
        self.assertEqual(model.count('CacheLayout::HeadContiguousPrefix =>'), 1)

    def test_exact_independent_test_inventory_and_required_boundaries(self):
        storage_tests = (patch.HERE / 'storage_tests.rs').read_text(encoding='utf-8')
        model_tests = (patch.HERE / 'model_tests.rs').read_text(encoding='utf-8')
        names = []
        for prefix, text in zip(patch.TEST_FILTERS, (storage_tests, model_tests)):
            self.assertNotIn('#[ignore', text)
            names.extend(prefix + n for n in re.findall(r'#\[test\]\s+fn\s+(\w+)\(', text))
        self.assertEqual(len(names), 12)
        self.assertEqual(sorted(names), sorted(patch.TEST_NAMES))
        self.assertTrue(all(sum(f in n for f in patch.TEST_FILTERS) == 1 for n in names))
        for required in ['Simd::Scalar, Simd::Auto, Simd::Avx2, Simd::Avx512',
                         '[6544, 6545, 6655, 6656, 6657, 16383, 16384]',
                         '0x80000000', '0x7fc01234', '0x7fa04321',
                         'bits_equal(&actual, &expected)', 'catch_unwind', 'drop(scratch)']:
            self.assertIn(required, storage_tests)
        self.assertEqual(self.result['src/head_contiguous_prefix_model_tests.rs'],
                         (patch.HERE / 'model_tests.rs').read_bytes())

    def test_changed_baseline_and_ambiguous_markers_reject(self):
        for name in patch.CHANGED_SOURCE_NAMES:
            changed = dict(self.original); changed[name] += b'\n'
            with self.assertRaises(ValueError): patch.make_sources(changed)
        for name in patch.ADDED_SOURCE_NAMES:
            changed = dict(self.original); changed[name] = b'collision'
            with self.assertRaises(ValueError): patch.make_sources(changed)
        for text in ['absent', 'anchor anchor']:
            with self.assertRaises(ValueError): patch.once(text, 'anchor', 'replacement')
        with self.assertRaises(ValueError):
            patch.make_patch(self.kernel, self.template + '\n// GENERATED_FIXED_HELPERS')


if __name__ == '__main__':
    unittest.main()
