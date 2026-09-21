"""Exact-anchor copy patch. No live file is written by this module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNELS_SHA = "19f1f18e164fffcab63dd1d747aae76f3a4a75dfcc15edebacd46b5dc32f7f16"
START = "#[allow(clippy::too_many_arguments)]\npub fn attention_with_simd("
END = "\n/// A flash-style CPU prefill:"
DISPATCH_ANCHOR = "    let dot = dot_kernel(selected);"
DISPATCH = '''    #[cfg(target_arch = "x86_64")]
    if query_len == 1 && head_dim == 64 && selected == Simd::Avx2 {
        // SAFETY: Runtime features and complete shapes were checked above.
        unsafe { attention64_candidate::attention(q, k, v, n_heads, query_offset,
            image_start, image_end, sinks, output); }
        return;
    }
'''

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact source anchor: {old!r}")
    return text.replace(old, new)

def make_patch(original, template, tests):
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError("Original attention boundary missing/ambiguous")
    start = original.index(START)
    end = original.index(END, start)
    function = original[start:end]
    head_start = "            let query = qh / n_heads;"
    head_end = "\n        });\n}\n"
    if function.count(head_start) != 1 or not function.endswith(head_end):
        raise ValueError("Original per-head body boundary changed")
    body = function[function.index(head_start):-len(head_end)]
    body = once(body, "dot(qvec, &k[begin..begin + head_dim])", "dot64(qvec, &k[begin..begin + head_dim])")
    body = once(body, "axpy(probability, &v[begin..begin + head_dim], out);",
                "axpy64(probability, &v[begin..begin + head_dim], out);")
    candidate = once(template, "        // GENERATED_ORIGINAL_HEAD_BODY", body)
    patched_function = once(function, DISPATCH_ANCHOR, DISPATCH + DISPATCH_ANCHOR)
    oracle = once(function, "pub fn attention_with_simd(", "fn attention_original_for_test(")
    candidate = once(candidate, '#[cfg(test)]\n#[path = "attention64_tests.rs"]\nmod tests;',
                     '#[cfg(test)]\nmod tests {\n' + tests + '\n}\n')
    # Inline the whole experiment in the sole changed archived source file.
    module = '\n#[cfg(target_arch = "x86_64")]\nmod attention64_candidate {\n' + candidate + '\n}\n'
    patched = original[:start] + patched_function + original[end:] + module
    patched += "\n#[cfg(test)]\n" + oracle
    return patched, candidate, function, body
