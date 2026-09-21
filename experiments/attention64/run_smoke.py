"""One authorized copied-CLI smoke trace, compared with the saved CPU control."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from capture import ROOT, read, sha, write, need, load, closure

directory=ROOT/'artifacts/diagnostics/attention64-v2'
prep_sha='8cb319bc4cf50de8fce352b9fcd8a35920dec3b1c1034f917084ab079b84d553'
build_sha=sys.argv[1]
directory,prep=load(SimpleNamespace(prepared=directory,preparation_sha256=prep_sha))
build=read(directory/'build.json',build_sha)
need(build['preparation_sha256']==prep_sha,'Wrong build preparation')
operators_path=directory/'operators.json';operators_sha=sha(operators_path)
operators=read(operators_path,operators_sha)
need(operators['status']=='passed' and operators['build_sha256']==build_sha,'Operator gate incomplete')
binary=directory/build['executables']['cli']['path'];binary_sha=build['executables']['cli']['sha256']
reference=ROOT/'artifacts/diagnostics/stage-timing-windows-v1/trace.safetensors'
reference_sha='e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309'
fixture=ROOT/'artifacts/reference/smoke-fp32/trace.safetensors'
fixture_sha='30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4'
script=Path(__file__).resolve();script_sha=sha(script)
bound={binary:binary_sha,reference:reference_sha,fixture:fixture_sha,script:script_sha,
       directory/'build.json':build_sha,operators_path:operators_sha,
       reference.with_suffix('.json'):'694c726f3d259df9237939f216a1022ed581d082c4f7669aff867d44b4d8f4a9'}
need(all(sha(p)==h for p,h in bound.items()),'Changed input')
output=directory/'smoke';output.mkdir(exist_ok=False)
trace=output/'trace.safetensors'
cmd=[str(binary),'--model',str(ROOT/'artifacts/model'),'--threads','4','--backend','avx2',
     '--precision','fp32','--batch-size','1','--cache-layout','expanded','--weight-layout','unpacked',
     'trace','--fixture',str(fixture),'--output',str(trace),'--max-new-tokens','17']
write(output/'invocation.json',{'command':cmd,'build_sha256':build_sha,'binary_sha256':binary_sha,
    'source_sha256':script_sha,'fixture_sha256':fixture_sha,'reference_sha256':reference_sha})
with (output/'stdout.log').open('x') as out,(output/'stderr.log').open('x') as err:
    code=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),stdout=out,stderr=err).returncode
report={'kind':'attention64-smoke-v1','exit_code':code,'build_sha256':build_sha,'binary_sha256':binary_sha,
        'operators_sha256':operators_sha,'reference_sha256':reference_sha,'fixture_sha256':fixture_sha,
        'script_sha256':script_sha,'status':'failed','performance_claim':False,'gpu_qualification_claim':False}
try:
    need(code==0,'CLI failed')
    raw=trace.read_bytes();trace_sha=hashlib.sha256(raw).hexdigest()
    header_len=int.from_bytes(raw[:8],'little');header=json.loads(raw[8:8+header_len])
    count=len([n for n in header if n!='__metadata__'])
    result=json.loads(trace.with_suffix('.json').read_bytes())
    control=read(reference.with_suffix('.json'),bound[reference.with_suffix('.json')])
    fields=['token_ids','text','finish_reason','output_tokens','input_tokens','precision','teacher_forced',
            'width','height','backend','cache_layout','weight_layout','packed_weight_bytes']
    need(all(result.get(f)==control[f] for f in fields),'Result differs from saved CPU control')
    need(trace_sha==reference_sha and count==1904,'Trace not byte-exact')
    need(len(result['token_ids'])==17 and result['teacher_forced'] is True,'Wrong teacher trace')
    closure(directory,prep);need(all(sha(p)==h for p,h in bound.items()),'Input/source changed')
    report.update(status='bit_exact',trace_sha256=trace_sha,tensor_count=count,teacher_token_count=17,
                  compared_result_fields=fields,source_and_input_closure=True)
finally:
    report['output_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write(output/'report.json',report)
print(json.dumps({'status':report['status'],'report_sha256':sha(output/'report.json')}))
