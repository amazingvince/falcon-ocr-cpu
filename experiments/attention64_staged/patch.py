"""One compact AVX2 PV scheduling change in the frozen combined-attention copy.

Import performs no I/O. make_sources reads this experiment's two Rust templates;
it never writes a project, imports a previous mutable patch, or invokes tools.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = ROOT / 'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
BASELINE_BUILD_SHA = '68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0'
BASELINE_ARCHIVE_SHA = 'f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3'
KERNELS_SHA = 'ad812805e03b536de0408d6f0dd21d3e944d6d752bc510331c59da44fe80fc5c'
CHANGED_SOURCE_NAMES = ('src/kernels.rs',)
ADDED_SOURCE_NAMES = ()
SOURCE_FILES = tuple('experiments/attention64_staged/' + name for name in
                     ('candidate.rs', 'patch.py', 'tests.rs', 'test_source.py', 'README.md', 'capture.py'))
INPUT_PINS = {}
TEST_FILTER = 'kernels::attention64_'
TEST_FILTERS = ('kernels::attention64_candidate::tests::',
                'kernels::attention64_staged::tests::')
OLD_TEST_NAMES = (
    'vectors_preserve_four_accumulators_and_fma',
    'causal_tile_tails_are_bit_exact',
    'image_and_causal_boundaries_are_bit_exact',
    'extreme_logits_sinks_and_long_context_are_bit_exact',
    'auto_and_one_thread_are_bit_exact',
    'other_shapes_backends_and_prefill_stay_on_original_path',
    'compact_zero_full_and_tile_crossovers_are_bit_exact',
    'compact_image_and_generated_boundaries_are_bit_exact',
    'compact_real_prefix_full_context_and_extremes_are_bit_exact',
    'compact_gqa_repeats_and_auto_are_bit_exact',
    'compact_other_shapes_and_prefill_stay_unchanged',
)
NEW_TEST_NAMES = (
    'pv_tile_preserves_rescale_fma_order_and_unaligned_guards',
    'staged_tiles_preserve_increasing_maxima_and_cancellation',
    'negative_values_signed_zeros_and_sink_extremes_are_exact',
    'paired_heads_masks_and_prefix_crossovers_are_exact',
    'real_prefix_full_context_and_partial_final_tiles_are_exact',
    'public_validation_and_unchanged_paths_are_preserved',
)
TEST_NAMES = tuple(TEST_FILTERS[0] + name for name in OLD_TEST_NAMES) + tuple(
    TEST_FILTERS[1] + name for name in NEW_TEST_NAMES)
FUNCTION_START = '#[allow(clippy::too_many_arguments)]\npub fn attention_compact_with_simd('
FUNCTION_END = '\n#[allow(clippy::too_many_arguments)]\nfn attention_gemm_compact('
MODULE_START = '\n#[cfg(target_arch = "x86_64")]\nmod attention64_staged {\n'
OLD_CALL = 'unsafe { attention64_candidate::compact(q, prefix_k, generated_k, v,'
NEW_CALL = 'unsafe { attention64_staged::compact(q, prefix_k, generated_k, v,'
OLD_PV = '''                for value in out.iter_mut() {
                    *value *= rescale;
                }
                denominator *= rescale;
                for (j, logit) in logits[..len].iter().enumerate() {
                    let probability = (*logit - new_max).exp();
                    denominator += probability;
                    let begin = (start + j) * kv_width + kv_head * head_dim;
                    axpy64(probability, &v[begin..begin + head_dim], out);
                }'''
NEW_PV = '''                denominator *= rescale;
                for logit in logits[..len].iter_mut() {
                    let probability = (*logit - new_max).exp();
                    denominator += probability;
                    *logit = probability;
                }
                let first = start * kv_width + kv_head * head_dim;
                pv_tile(&logits[..len], v, first, kv_width, rescale, out);'''


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one exact source anchor: {old!r}')
    return text.replace(old, new)


def between(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Missing or ambiguous extraction boundary')
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def make_patch(original, template, tests):
    """Return patched kernels and exact extracted source pieces for auditing."""
    if hashlib.sha256(original.encode('utf-8')).hexdigest() != KERNELS_SHA:
        raise ValueError('Expected exact frozen combined-attention kernels')
    function = between(original, FUNCTION_START, FUNCTION_END)
    changed = once(function, OLD_CALL, NEW_CALL)
    frozen_module = original[original.index('mod attention64_candidate {'):]
    dot = between(frozen_module, '#[inline(always)]\nunsafe fn dot64(',
                  '#[inline(always)]\nunsafe fn axpy64(')
    compact_start = '#[allow(clippy::too_many_arguments)]\npub(super) unsafe fn compact('
    head_start = '#[target_feature(enable="avx2,fma")]\n#[allow(clippy::too_many_arguments)]\nunsafe fn compact_head('
    wrapper = between(frozen_module, compact_start, head_start)
    # The first cfg(test) after compact_head is its untouched old test module.
    head_begin = frozen_module.index(head_start)
    head_end = frozen_module.index('\n#[cfg(test)]\nmod tests {', head_begin)
    head = frozen_module[head_begin:head_end]
    staged_head = once(head, OLD_PV, NEW_PV)
    candidate = once(template, '// GENERATED_WRAPPER', wrapper)
    candidate = once(candidate, '// GENERATED_HEAD', staged_head)
    candidate = once(candidate, '// GENERATED_DOT64', dot)
    candidate = once(candidate, '    // GENERATED_TESTS', tests)
    oracle = once(function, 'pub fn attention_compact_with_simd(',
                  'fn attention_compact_combined_for_staged_test(')
    patched = once(original, function, changed)
    patched += MODULE_START + candidate + '\n}\n'
    patched += '\n#[cfg(test)]\n' + oracle
    return patched, {'function': function, 'wrapper': wrapper, 'head': head,
                     'staged_head': staged_head, 'dot64': dot, 'candidate': candidate}


def make_sources(original: dict[str, bytes]) -> dict[str, bytes]:
    if 'src/kernels.rs' not in original:
        raise ValueError('Source inventory has no kernels')
    template = (HERE / 'candidate.rs').read_text(encoding='utf-8')
    tests = (HERE / 'tests.rs').read_text(encoding='utf-8')
    patched, _ = make_patch(original['src/kernels.rs'].decode('utf-8'), template, tests)
    result = dict(original)
    result['src/kernels.rs'] = patched.encode('utf-8')
    return result
