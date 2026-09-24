"""Exact-anchor copied-project patch; no import-time I/O or live source edits."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASELINE = ROOT / 'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
BASELINE_BUILD_SHA = '68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0'
BASELINE_ARCHIVE_SHA = 'f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3'
KERNELS_SHA = 'ad812805e03b536de0408d6f0dd21d3e944d6d752bc510331c59da44fe80fc5c'
BASELINE_SOURCE_PINS = {
    'src/kernels.rs': KERNELS_SHA,
    'src/config.rs': 'aaa40cc566e4f3d585beef097b560831d8f38f17490d6038f6d605954cf1d1ee',
    'src/lib.rs': 'f6a853cbe67e1ce0bef47dcc61afe709d347cd855b4bab266f48d89760c469c7',
    'src/model.rs': 'a048d6d618fc800c8331cb1d8eae51ced65d813d93784391b65c77c02cfd2fe6',
}
CHANGED_SOURCE_NAMES = tuple(BASELINE_SOURCE_PINS)
ADDED_SOURCE_NAMES = ('src/temporal_candidate.rs', 'src/temporal_model_tests.rs')
REUSED_SOURCE_PINS = {
    'temporal_candidate.rs': 'd636e7fb065b50fde80bf905cf245a69fe60d2b2a8e45361ad6eed3bed1f7fc8',
    'temporal_model_tests.rs': '51c7fe618cbb2760fcc3efd1ba329d36a107b7888ce7fdcafdb75bc5c12cbfc2',
}
SOURCE_FILES = tuple('experiments/attention64_temporal/' + name for name in (
    'candidate.rs', 'tests.rs', 'patch.py', 'test_source.py', 'README.md',
    'temporal_candidate.rs', 'temporal_model_tests.rs',
))
TEST_FILTERS = (
    'kernels::temporal64::tests::',
    'temporal_candidate::tests::',
    'model::temporal_candidate_tests::',
)
TEST_NAMES = tuple(TEST_FILTERS[0] + name for name in (
    'split_dot_preserves_four_accumulators_unaligned_halves_and_cancellation',
    'split_attention_zero_full_prefix_and_tile_crossovers_match',
    'split_attention_image_boundaries_and_extreme_sinks_match',
    'split_attention_real_prefix_and_full_context_match',
    'split_attention_rejects_bad_shapes_before_dispatch',
)) + tuple(TEST_FILTERS[1] + name for name in (
    'storage_bits_including_signed_zero_and_nan_payload',
    'invalid_duplicate_rejected_before_mutation',
    'unsupported_dimensions_partial_prefix_continuation_and_capacity',
    'unchanged_prefill_and_single_decode_all_supported_backends',
    'decode_masks_key_tiles_long_tails_and_extreme_sinks',
    'reserved_owned_payload_and_append_capacity_stability',
)) + tuple(TEST_FILTERS[2] + name for name in (
    'model_cache_dispatch_prefill_decode_matches_compact',
    'model_candidate_is_explicit_default_unchanged',
))
START = 'pub fn attention_compact_with_simd('
END = '\n#[allow(clippy::too_many_arguments)]\nfn attention_gemm_compact('
KEY_DOT_BLOCK = '''                    let kvec = if key < prefix_len {
                        let begin = key * query_width + head * head_dim;
                        &prefix_k[begin..begin + head_dim]
                    } else {
                        let begin = (key - prefix_len) * kv_width + kv_head * head_dim;
                        &generated_k[begin..begin + head_dim]
                    };
                    *logit = dot64(qvec, kvec) * scale;'''
SPLIT_KEY_DOT_BLOCK = '''                    let score = if key < prefix_len {
                        let temporal = (key * n_kv_heads + kv_head) * 32;
                        let spatial = (key * n_heads + head) * 32;
                        dot64_split(qvec, &temporal_k[temporal..temporal + 32],
                                    &spatial_k[spatial..spatial + 32])
                    } else {
                        let begin = (key - prefix_len) * kv_width + kv_head * head_dim;
                        dot64(qvec, &generated_k[begin..begin + head_dim])
                    };
                    *logit = score * scale;'''
DISPATCH = '''    #[cfg(target_arch = "x86_64")]
    if selected == Simd::Avx2 {
        // SAFETY: Fixed dimensions, full slices and AVX2/FMA checked above.
        unsafe { temporal64::attention(q, temporal_k,
            spatial_k, generated_k, v, prefix_len, n_heads, n_kv_heads,
            query_offset, image_start, image_end, sinks, output); }
        return;
    }
'''


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected one exact source anchor: ' + repr(old[:100]))
    return text.replace(old, new)


def compact_function(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('Compact function boundary changed')
    start = source.index(START)
    return source[start:source.index(END, start)]


def generic_adapter(source):
    """Preserve the historical reconstruction path; exclude new compact dispatch."""
    result = compact_function(source)
    result = once(result, START, 'pub(crate) fn attention_temporal_candidate_with_simd(')
    result = once(result, '    prefix_k: &[f32],', '    temporal_k: &[f32],\n    spatial_k: &[f32],')
    result = once(result, ') {\n    assert!(', ') {\n    assert_eq!((n_heads,n_kv_heads,head_dim),(16,8,64),"temporal candidate fixed dimensions");\n    assert_eq!(query_len,1,"temporal candidate supports only one query");\n    assert!(')
    result = once(result, '        prefix_k.len(),\n        elements(prefix_len, query_width),\n        "compact attention prefix K shape"',
                  '        temporal_k.len(),\n        elements(prefix_len, 256),\n        "temporal attention prefix K shape"')
    result = once(result, '    assert_eq!(\n        generated_k.len(),',
                  '    assert_eq!(spatial_k.len(),elements(prefix_len,512),"spatial attention prefix K shape");\n    assert_eq!(\n        generated_k.len(),')
    # Both inherited GEMM and compact fixed64 branches reference full prefix K.
    # query_len==1 is checked above; retain the validated generic body beneath.
    begin = result.index('    if query_len >= 4 && selected != Simd::Scalar {')
    end = result.index('    let dot = dot_kernel(selected);', begin)
    removed = result[begin:end]
    if removed.count('attention_gemm_compact(') != 1 or removed.count('attention64_candidate::compact(') != 1:
        raise ValueError('Unexpected inherited prefill/compact dispatch')
    result = result[:begin] + result[end:]
    result = once(result, '            let mut logits = [0.0_f32; TILE];',
                  '            let mut logits = [0.0_f32; TILE];\n            let mut reconstructed = [0.0_f32; 64];')
    result = once(result, '                        let begin = key * query_width + head * head_dim;\n                        &prefix_k[begin..begin + head_dim]',
                  '                        let temporal = (key * 8 + kv_head) * 32;\n                        let spatial = (key * 16 + head) * 32;\n                        reconstructed[..32].copy_from_slice(&temporal_k[temporal..temporal+32]);\n                        reconstructed[32..].copy_from_slice(&spatial_k[spatial..spatial+32]);\n                        &reconstructed[..]')
    if 'prefix_k' in result:
        raise ValueError('Unpatched full-prefix K reference')
    return result


def make_patch(original, template, tests):
    """Return patched kernels, appended candidate, and unchanged generic oracle."""
    if hashlib.sha256(original.encode('utf-8')).hexdigest() != KERNELS_SHA:
        raise ValueError('Expected frozen combined-attention kernels')
    generic = generic_adapter(original)
    adapter = once(generic, '    let dot = dot_kernel(selected);', DISPATCH + '    let dot = dot_kernel(selected);')
    oracle = once(generic, 'pub(crate) fn attention_temporal_candidate_with_simd(',
                  'fn attention_temporal_generic_for_test(')
    head_start = original.index('unsafe fn compact_head(')
    body_start = original.index('            let query = qh / n_heads;', head_start)
    head_end = original.index('\n#[cfg(test)]\nmod tests {', body_start)
    tail = original[body_start:head_end]
    if not tail.endswith('    }\n}\n'):
        raise ValueError('Compact head closing boundary changed')
    body = tail[:-len('    }\n}\n')]
    body = once(body, KEY_DOT_BLOCK, SPLIT_KEY_DOT_BLOCK)
    helper_start = original.index('#[inline(always)]\nunsafe fn dot64(')
    helper_end = original.index('\n#[allow(clippy::too_many_arguments)]\npub(super) unsafe fn compact(', helper_start)
    helpers = original[helper_start:helper_end]
    if helpers.count('unsafe fn dot64(') != 1 or helpers.count('unsafe fn axpy64(') != 1:
        raise ValueError('Frozen fixed64 helper inventory changed')
    candidate = once(template, '        // GENERATED_ORIGINAL_COMPACT_HEAD_BODY', body.rstrip('\n'))
    candidate = once(candidate, '// GENERATED_FIXED_HELPERS', helpers)
    candidate = once(candidate, '    // GENERATED_TESTS', tests)
    patched = original + '\n#[allow(clippy::too_many_arguments)]\n' + adapter
    patched += '\n#[cfg(target_arch = "x86_64")]\nmod temporal64 {\n' + candidate + '\n}\n'
    patched += '\n#[cfg(test)]\n#[allow(clippy::too_many_arguments)]\n' + oracle
    return patched, candidate, generic


def patch_model(model):
    """Historical storage integration, retaining separate Compact/Expanded paths."""
    model = once(model, 'enum LayerCache {\n', 'enum LayerCache {\n    Temporal(crate::temporal_candidate::TemporalCache),\n')
    begin = model.index('impl LayerCache {')
    end = model.index('\n/// Store one original GQA head', begin)
    block = model[begin:end]
    block = once(block, 'fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) {\n        match self {',
                 'fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) -> Result<()> {\n        match self {\n            Self::Temporal(cache) => cache.append(k,v,offset)?,')
    block = once(block, '        }\n    }\n\n    #[allow', '        }\n        Ok(())\n    }\n\n    #[allow')
    block = once(block, '        q: &[f32],\n', '        q: &[f32],\n        current_expanded_k: &[f32],\n')
    block = once(block, '    ) {\n        match self {', '    ) -> Result<()> {\n        match self {\n            Self::Temporal(cache) => cache.attention(q,current_expanded_k,rows,total_len,offset,image_start,image_end,sinks,output,simd)?,')
    ending = '        }\n    }\n}\n'
    if not block.endswith(ending):
        raise ValueError('Unexpected LayerCache block ending')
    block = block[:-len(ending)] + '        }\n        Ok(())\n    }\n}\n'
    model = model[:begin] + block + model[end:]
    model = once(model, 'cache.append(&work.k, &work.v, offset, c);', 'cache.append(&work.k, &work.v, offset, c)?;')
    model = once(model, 'cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c);',
                 'cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c)?;')
    model = once(model, '            cache.attention(\n                &work.q,', '            cache.attention(\n                &work.q,\n                &work.k,')
    model = once(model, '                cache.attention(\n                    &work.q[range.clone()],',
                 '                cache.attention(\n                    &work.q[range.clone()],\n                    &work.k[range.clone()],')
    for marker in ['            cache.attention(', '                cache.attention(']:
        begin = model.index(marker)
        end = model.index('\n            }' if marker.startswith('                ') else '\n            if trace.enabled()', begin)
        call = model[begin:end]
        if not call.endswith(');'):
            raise ValueError('Unexpected attention call close')
        model = model[:begin] + call[:-2] + ')?;' + model[end:]
    model = once(model, '            layers.push(match cache_layout {',
                 '            layers.push(match cache_layout {\n                CacheLayout::TemporalCandidate => LayerCache::Temporal(crate::temporal_candidate::TemporalCache::new(prefix_len,capacity,c.n_heads,c.n_kv_heads,c.head_dim)?),')
    return model + '\n#[cfg(test)]\n#[path = "temporal_model_tests.rs"]\nmod temporal_candidate_tests;\n'


def make_sources(original: dict[str, bytes]) -> dict[str, bytes]:
    """Preserve all original members; change exactly four and add exactly two.

    The caller separately binds the entire input archive/build manifest. Reads
    only this experiment's seven source files; no copy, build or execution.
    """
    for name, digest in BASELINE_SOURCE_PINS.items():
        if hashlib.sha256(original[name]).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    if any(name in original for name in ADDED_SOURCE_NAMES):
        raise ValueError('Candidate module already present')
    result = dict(original)
    source = {name: original[name].decode('utf-8') for name in CHANGED_SOURCE_NAMES}
    kernel, _, _ = make_patch(source['src/kernels.rs'],
        (HERE / 'candidate.rs').read_text(encoding='utf-8'), (HERE / 'tests.rs').read_text(encoding='utf-8'))
    source['src/kernels.rs'] = kernel
    source['src/config.rs'] = once(source['src/config.rs'], '    Expanded,\n    Compact,\n}',
        '    Expanded,\n    Compact,\n    /// Isolated experimental prefix temporal-key sharing.\n    TemporalCandidate,\n}')
    source['src/lib.rs'] += '\nmod temporal_candidate;\n'
    source['src/model.rs'] = patch_model(source['src/model.rs'])
    result.update({name: text.encode('utf-8') for name, text in source.items()})
    for name, digest in REUSED_SOURCE_PINS.items():
        raw = (HERE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Preserved historical module changed: ' + name)
        result['src/' + name] = raw
    changed = {name for name in original if original[name] != result[name]}
    added = set(result) - set(original)
    if changed != set(CHANGED_SOURCE_NAMES) or added != set(ADDED_SOURCE_NAMES):
        raise ValueError('Undeclared source delta')
    return result
