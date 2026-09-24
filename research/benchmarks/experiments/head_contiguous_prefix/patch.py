"""Pinned copied-project patch. Importing does not read/write source or build."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
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
ADDED_SOURCE_NAMES = ('src/head_contiguous_prefix.rs', 'src/head_contiguous_prefix_model_tests.rs')
SOURCE_FILES = tuple('experiments/head_contiguous_prefix/' + name for name in (
    'candidate.rs', 'head_contiguous_prefix.rs', 'patch.py', 'README.md',
    'storage_tests.rs', 'model_tests.rs', 'test_source.py',
))
TEST_FILTERS = ('head_contiguous_prefix::tests::', 'model::head_contiguous_prefix_tests::')
TEST_NAMES = (
    'head_contiguous_prefix::tests::storage_full_head_order_and_special_bits',
    'head_contiguous_prefix::tests::duplicate_bit_failures_leave_every_buffer_unchanged',
    'head_contiguous_prefix::tests::invalid_sizes_offsets_and_overflow_reject_before_mutation',
    'head_contiguous_prefix::tests::reserved_capacity_and_addresses_stay_fixed_through_appends',
    'head_contiguous_prefix::tests::reusable_prefill_scratch_has_compact_order_and_transactional_errors',
    'head_contiguous_prefix::tests::cache_attention_rejects_wrong_intervals_and_retained_scratch',
    'head_contiguous_prefix::tests::operator_short_prefix_zero_full_and_tile_crossings_all_supported_backends',
    'head_contiguous_prefix::tests::operator_fullpage_crossing_tile_and_16384_context_all_supported_backends',
    'head_contiguous_prefix::tests::unchanged_prefill_and_storage_decode_dispatch_match_compact',
    'head_contiguous_prefix::tests::operator_invalid_shapes_preserve_output_before_dispatch',
    'model::head_contiguous_prefix_tests::model_head_prefix_prefill_then_decode_matches_unchanged_compact',
    'model::head_contiguous_prefix_tests::model_head_prefix_enum_is_explicit_and_invalid_sessions_reject',
)

START = 'pub fn attention_compact_with_simd('
END = '\n#[allow(clippy::too_many_arguments)]\nfn attention_gemm_compact('
OLD_K_ADDRESS = '                        let begin = key * query_width + head * head_dim;'
NEW_K_ADDRESS = '                        let begin = (head * prefix_len + key) * head_dim;'
OLD_V_ADDRESS = '''                    let begin = (start + j) * kv_width + kv_head * head_dim;
                    {axpy}(probability, &v[begin..begin + head_dim], out);'''
NEW_V_ADDRESS = '''                    let key = start + j;
                    let vvec = if key < prefix_len {{
                        let begin = (kv_head * prefix_len + key) * head_dim;
                        &prefix_v[begin..begin + head_dim]
                    }} else {{
                        let begin = (key - prefix_len) * kv_width + kv_head * head_dim;
                        &generated_v[begin..begin + head_dim]
                    }};
                    {axpy}(probability, vvec, out);'''
DISPATCH = '''    #[cfg(target_arch = "x86_64")]
    if selected == Simd::Avx2 {
        // SAFETY: Complete fixed dimensions/slices and AVX2/FMA checked above.
        unsafe { head_prefix64::attention(q, prefix_k, prefix_v, generated_k, generated_v,
            prefix_len, n_heads, n_kv_heads, query_offset, image_start, image_end,
            sinks, output); }
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
    begin = source.index(START)
    return source[begin:source.index(END, begin)]


def change_addresses(body, axpy):
    body = once(body, OLD_K_ADDRESS, NEW_K_ADDRESS)
    return once(body, OLD_V_ADDRESS.format(axpy=axpy), NEW_V_ADDRESS.format(axpy=axpy))


def generic_adapter(source):
    """Keep old scalar/AVX512 math, tiles and key order; change addresses only."""
    result = compact_function(source)
    result = once(result, START, 'pub(crate) fn attention_head_contiguous_prefix_with_simd(')
    result = once(result, '    prefix_k: &[f32],\n    generated_k: &[f32],\n    v: &[f32],',
        '    prefix_k: &[f32],\n    prefix_v: &[f32],\n    generated_k: &[f32],\n    generated_v: &[f32],')
    result = once(result, ') {\n    assert!(', ') {\n    assert_eq!((n_heads,n_kv_heads,head_dim),(16,8,64),"head-contiguous prefix fixed dimensions");\n    assert_eq!(query_len,1,"head-contiguous decode supports only one query");\n    assert!(')
    result = once(result, '''    assert_eq!(
        v.len(),
        elements(total_len, kv_width),
        "compact attention V shape"
    );''', '''    assert_eq!(prefix_v.len(), elements(prefix_len, kv_width), "head-contiguous prefix V shape");
    assert_eq!(generated_v.len(), elements(total_len - prefix_len, kv_width), "head-contiguous generated V shape");''')
    begin = result.index('    if query_len >= 4 && selected != Simd::Scalar {')
    end = result.index('    let dot = dot_kernel(selected);', begin)
    removed = result[begin:end]
    if removed.count('attention_gemm_compact(') != 1 or removed.count('attention64_candidate::compact(') != 1:
        raise ValueError('Unexpected inherited prefill/compact dispatch')
    result = result[:begin] + result[end:]
    return change_addresses(result, 'axpy')


def make_patch(original, template):
    """Append new adapter/module, retaining every baseline kernel byte."""
    if hashlib.sha256(original.encode('utf-8')).hexdigest() != KERNELS_SHA:
        raise ValueError('Expected frozen combined kernels')
    generic = generic_adapter(original)
    adapter = once(generic, '    let dot = dot_kernel(selected);', DISPATCH + '    let dot = dot_kernel(selected);')
    head_start = original.index('unsafe fn compact_head(')
    body_start = original.index('            let query = qh / n_heads;', head_start)
    head_end = original.index('\n#[cfg(test)]\nmod tests {', body_start)
    tail = original[body_start:head_end]
    if not tail.endswith('    }\n}\n'):
        raise ValueError('Compact head closing boundary changed')
    body = change_addresses(tail[:-len('    }\n}\n')], 'axpy64')
    helper_start = original.index('#[inline(always)]\nunsafe fn dot64(')
    helper_end = original.index('\n#[allow(clippy::too_many_arguments)]\npub(super) unsafe fn compact(', helper_start)
    helpers = original[helper_start:helper_end]
    if helpers.count('unsafe fn dot64(') != 1 or helpers.count('unsafe fn axpy64(') != 1:
        raise ValueError('Fixed64 helper inventory changed')
    candidate = once(template, '        // GENERATED_ORIGINAL_COMPACT_HEAD_BODY', body.rstrip('\n'))
    candidate = once(candidate, '// GENERATED_FIXED_HELPERS', helpers)
    patched = original + '\n#[allow(clippy::too_many_arguments)]\n' + adapter
    patched += '\n#[cfg(target_arch = "x86_64")]\nmod head_prefix64 {\n' + candidate + '\n}\n'
    return patched, candidate, generic


def patch_model(model):
    model = once(model, 'enum LayerCache {\n', 'enum LayerCache {\n    HeadContiguousPrefix(crate::head_contiguous_prefix::HeadContiguousPrefixCache),\n')
    begin = model.index('impl LayerCache {')
    end = model.index('\n/// Store one original GQA head', begin)
    block = model[begin:end]
    block = once(block, 'fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) {\n        match self {',
        'fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) -> Result<()> {\n        match self {\n            Self::HeadContiguousPrefix(cache) => cache.append(k,v,offset)?,')
    block = once(block, '        }\n    }\n\n    #[allow', '        }\n        Ok(())\n    }\n\n    #[allow')
    block = once(block, '        q: &[f32],\n', '        q: &[f32],\n        current_expanded_k: &[f32],\n        prefill_compact_v: &[f32],\n')
    block = once(block, '    ) {\n        match self {',
        '    ) -> Result<()> {\n        match self {\n            Self::HeadContiguousPrefix(cache) => cache.attention(q,current_expanded_k,prefill_compact_v,rows,total_len,offset,image_start,image_end,sinks,output,simd)?,')
    ending = '        }\n    }\n}\n'
    if not block.endswith(ending):
        raise ValueError('Unexpected LayerCache block ending')
    block = block[:-len(ending)] + '        }\n        Ok(())\n    }\n}\n'
    model = model[:begin] + block + model[end:]
    # One transient token-major V scratch only during candidate full prefill.
    # Capacity is reused across layers; local destruction charges its release.
    model = once(model, '''        let offset = session.len;
        let simd = session.simd;
        let work = &mut session.workspace;''', '''        let offset = session.len;
        let head_contiguous_prefill = offset == 0
            && matches!(session.layers.first(), Some(LayerCache::HeadContiguousPrefix(_)));
        let mut prefill_compact_v = Vec::new();
        if head_contiguous_prefill {
            let count = rows.checked_mul(kdim).context("prefill compact V size overflow")?;
            prefill_compact_v.try_reserve_exact(count).context("allocate transient prefill compact V")?;
        }
        let simd = session.simd;
        let work = &mut session.workspace;''')
    model = once(model, 'cache.append(&work.k, &work.v, offset, c);',
        '''cache.append(&work.k, &work.v, offset, c)?;
            if head_contiguous_prefill {
                crate::head_contiguous_prefix::pack_prefill_values(&work.v, &mut prefill_compact_v)?;
            }''')
    model = once(model, 'cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c);',
        'cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c)?;')
    model = once(model, '            cache.attention(\n                &work.q,',
        '            cache.attention(\n                &work.q,\n                &work.k,\n                &prefill_compact_v,')
    model = once(model, '                cache.attention(\n                    &work.q[range.clone()],',
        '                cache.attention(\n                    &work.q[range.clone()],\n                    &work.k[range.clone()],\n                    &[],')
    for marker in ['            cache.attention(', '                cache.attention(']:
        begin = model.index(marker)
        end = model.index('\n            }' if marker.startswith('                ') else '\n            if trace.enabled()', begin)
        call = model[begin:end]
        if not call.endswith(');'):
            raise ValueError('Unexpected attention call close')
        model = model[:begin] + call[:-2] + ')?;' + model[end:]
    model = once(model, '            layers.push(match cache_layout {',
        '            layers.push(match cache_layout {\n                CacheLayout::HeadContiguousPrefix => LayerCache::HeadContiguousPrefix(crate::head_contiguous_prefix::HeadContiguousPrefixCache::new(prefix_len,capacity,c.n_heads,c.n_kv_heads,c.head_dim)?),')
    return model + '\n#[cfg(test)]\n#[path = "head_contiguous_prefix_model_tests.rs"]\nmod head_contiguous_prefix_tests;\n'


def make_sources(original: dict[str, bytes]) -> dict[str, bytes]:
    """Change exactly four frozen members, add exactly two; no copying/building."""
    for name, digest in BASELINE_SOURCE_PINS.items():
        if hashlib.sha256(original[name]).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    if any(name in original for name in ADDED_SOURCE_NAMES):
        raise ValueError('Candidate module already present')
    result = dict(original)
    source = {name: original[name].decode('utf-8') for name in CHANGED_SOURCE_NAMES}
    source['src/kernels.rs'], _, _ = make_patch(source['src/kernels.rs'], (HERE / 'candidate.rs').read_text(encoding='utf-8'))
    source['src/config.rs'] = once(source['src/config.rs'], '    Expanded,\n    Compact,\n}',
        '    Expanded,\n    Compact,\n    /// Isolated physical prefix K/V reordering; arithmetic unchanged.\n    HeadContiguousPrefix,\n}')
    source['src/lib.rs'] += '\nmod head_contiguous_prefix;\n'
    source['src/model.rs'] = patch_model(source['src/model.rs'])
    result.update({name: text.encode('utf-8') for name, text in source.items()})
    storage = (HERE / 'head_contiguous_prefix.rs').read_text(encoding='utf-8')
    storage = once(storage, '    // GENERATED_STORAGE_TESTS', (HERE / 'storage_tests.rs').read_text(encoding='utf-8'))
    result['src/head_contiguous_prefix.rs'] = storage.encode('utf-8')
    result['src/head_contiguous_prefix_model_tests.rs'] = (HERE / 'model_tests.rs').read_bytes()
    if {name for name in original if original[name] != result[name]} != set(CHANGED_SOURCE_NAMES) or set(result) - set(original) != set(ADDED_SOURCE_NAMES):
        raise ValueError('Undeclared source delta')
    return result
