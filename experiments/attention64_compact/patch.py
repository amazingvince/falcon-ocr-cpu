"""Extend the frozen reviewed expanded patch in a copied project only."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/attention64'
PRIOR_PREP=ROOT/'artifacts/diagnostics/attention64-v2/preparation.json'
PRIOR_SHA='8cb319bc4cf50de8fce352b9fcd8a35920dec3b1c1034f917084ab079b84d553'
prior_bytes=PRIOR_PREP.read_bytes()
if hashlib.sha256(prior_bytes).hexdigest()!=PRIOR_SHA:raise ValueError('Prior preparation changed')
prior=json.loads(prior_bytes)
for n in ['patch.py','candidate.rs','tests.rs']:
    path=OLD/n
    if hashlib.sha256(path.read_bytes()).hexdigest()!=prior['experiment_source_sha256'][path.relative_to(ROOT).as_posix()]:
        raise ValueError('Prior source changed: '+n)
spec=importlib.util.spec_from_file_location('expanded_attention_patch',OLD/'patch.py')
expanded=importlib.util.module_from_spec(spec);spec.loader.exec_module(expanded)
KERNELS_SHA=expanded.KERNELS_SHA
START='#[allow(clippy::too_many_arguments)]\npub fn attention_compact_with_simd('
END='\n#[allow(clippy::too_many_arguments)]\nfn attention_gemm_compact('
DISPATCH='''    #[cfg(target_arch = "x86_64")]
    if query_len == 1 && head_dim == 64 && selected == Simd::Avx2 {
        // SAFETY: Original entry point checked complete shapes and AVX2/FMA.
        unsafe { attention64_candidate::compact(q, prefix_k, generated_k, v,
            prefix_len, n_heads, n_kv_heads, query_offset, image_start, image_end,
            sinks, output); }
        return;
    }
'''

def make_patch(original,template,tests):
    # Require the exact previously accepted helper/expanded-source identities.
    prep=prior
    for name in ['patch.py','candidate.rs','tests.rs']:
        path=OLD/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=prep['experiment_source_sha256'][path.relative_to(ROOT).as_posix()]:
            raise ValueError('Prior expanded source changed: '+name)
    if original.count(START)!=1 or original.count(END)!=1:raise ValueError('Compact function boundary changed')
    start=original.index(START);end=original.index(END,start);function=original[start:end]
    first='            let query = qh / n_heads;';last='\n        });\n}\n'
    if function.count(first)!=1 or not function.endswith(last):raise ValueError('Compact head boundary changed')
    body=function[function.index(first):-len(last)]
    body=expanded.once(body,'dot(qvec, kvec)','dot64(qvec, kvec)')
    body=expanded.once(body,'axpy(probability, &v[begin..begin + head_dim], out);',
                       'axpy64(probability, &v[begin..begin + head_dim], out);')
    compact=expanded.once(template,'        // GENERATED_ORIGINAL_COMPACT_HEAD_BODY',body)
    old_template=(OLD/'candidate.rs').read_text(encoding='utf-8')
    insertion='#[cfg(test)]\n#[path = "attention64_tests.rs"]\nmod tests;'
    combined=expanded.once(old_template,insertion,compact+'\n'+insertion)
    combined_tests=(OLD/'tests.rs').read_text(encoding='utf-8')+'\n'+tests
    patched,candidate,_,_=expanded.make_patch(original,combined,combined_tests)
    patched_function=expanded.once(function,expanded.DISPATCH_ANCHOR,DISPATCH+expanded.DISPATCH_ANCHOR)
    patched=expanded.once(patched,function,patched_function)
    oracle=expanded.once(function,'pub fn attention_compact_with_simd(', 'fn attention_compact_original_for_test(')
    patched+='\n#[cfg(test)]\n'+oracle
    return patched,candidate,function,body
