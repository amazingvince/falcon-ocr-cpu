"""One exact-anchor patch of the frozen combined-attention copy; no I/O on import."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASELINE = ROOT / 'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
BASELINE_BUILD_SHA = '68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0'
BASELINE_ARCHIVE_SHA = 'f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3'
KERNELS_SHA = 'ad812805e03b536de0408d6f0dd21d3e944d6d752bc510331c59da44fe80fc5c'
START = 'pub fn linear_with_simd('
END = '\n/// Normalize each contiguous `width`-element row in FP32.'
ANCHOR = '    if rows <= 8 {'
DISPATCH = '''    #[cfg(target_arch = "x86_64")]
    if rows == 1 && selected == Simd::Avx2
        && gemv_pair_candidate::supported_shape(in_dim, out_dim)
    {
        // SAFETY: Complete shapes and AVX2/FMA were checked above.
        unsafe { gemv_pair_candidate::linear(input, weight, out); }
        return;
    }
'''
MODULE_START = '\n#[cfg(target_arch = "x86_64")]\nmod gemv_pair_candidate {\n'
TEST_FILTER = 'kernels::gemv_pair_candidate::tests::'
TEST_NAMES = (
    'all_five_actual_shapes_every_output_bit_matches',
    'auto_and_single_thread_match_on_guarded_shape',
    'paired_reductions_preserve_cancellation_zero_and_scale_boundaries',
    'unaligned_slices_and_channel_block_boundaries_match',
    'other_shapes_batches_and_backends_keep_original_outputs',
    'shape_guard_is_exact_and_public_validation_precedes_dispatch',
    'pinned_real_cpu_operands_match_all_five_projections',
)
INPUT_PINS = {
    'artifacts/model/config.json': 'ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf',
    'artifacts/model/model.safetensors': '3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16',
    'artifacts/diagnostics/fp32-crossover-layer7-v1/rust/tensors.safetensors': '4972d70e6ec5c6c744745282141abb7768dbea6314970affa49603cd955e102d',
    'artifacts/cpu/smoke-trace-sinks-pairwise.safetensors': 'e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309',
}


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one exact source anchor: {old!r}')
    return text.replace(old, new)


def make_patch(original, template, tests):
    """Return (patched kernels, inlined candidate, unchanged linear oracle).

    The caller must supply UTF-8 source decoded from the pinned archive and
    preserve all other archive entries. This function does not read or write
    live files or prepare/build/execute any project.
    """
    if hashlib.sha256(original.encode('utf-8')).hexdigest() != KERNELS_SHA:
        raise ValueError('Expected exact frozen combined-attention kernels')
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError('Original linear function boundary missing/ambiguous')
    start = original.index(START)
    end = original.index(END, start)
    function = original[start:end]
    patched_function = once(function, ANCHOR, DISPATCH + ANCHOR)
    candidate = once(template, '    // GENERATED_TESTS', tests)
    oracle = once(function, START, 'fn linear_original_for_gemv_pair_test(')
    patched = original[:start] + patched_function + original[end:]
    patched += MODULE_START + candidate + '\n}\n'
    patched += '\n#[cfg(test)]\n' + oracle
    return patched, candidate, function
