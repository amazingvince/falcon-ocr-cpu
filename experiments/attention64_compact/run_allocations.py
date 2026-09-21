"""Run the unchanged, captured warmed-decode allocation integration test once."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from capture import ROOT,load,read,sha,need,write,closure

directory=ROOT/'artifacts/diagnostics/attention64-compact-v1'
prep_sha='1862072c940d9aa2821562f7d1b94e6d7493692b10502aa60dafd2e8ce9458ce'
build_sha=sys.argv[1]
directory,prep=load(SimpleNamespace(prepared=directory,preparation_sha256=prep_sha))
build=read(directory/'build.json',build_sha)
need(build['preparation_sha256']==prep_sha,'Wrong build')
smoke=directory/'smoke/report.json';smoke_sha=sha(smoke)
need(read(smoke,smoke_sha)['status']=='bit_exact','Smoke control did not pass')
binary=directory/build['executables']['allocations']['path'];binary_sha=build['executables']['allocations']['sha256']
testsource=directory/'project/tests/decode_allocations.rs';test_sha=prep['project_source_sha256']['tests/decode_allocations.rs']
script=Path(__file__).resolve();bound={binary:binary_sha,testsource:test_sha,script:sha(script),
    directory/'build.json':build_sha,smoke:smoke_sha}
need(all(sha(p)==h for p,h in bound.items()),'Changed allocation input')
output=directory/'allocations';output.mkdir(exist_ok=False)
name='warm_fp32_decode_has_no_heap_allocations'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
listing=subprocess.check_output([str(binary),'--list'],text=True,env=env)
need(listing.splitlines().count(name+': test')==1,'Wrong test binary')
command=[str(binary),name,'--exact','--ignored','--nocapture','--test-threads','1']
write(output/'invocation.json',{'command':command,'cwd':str(ROOT),'build_sha256':build_sha,
    'binary_sha256':binary_sha,'test_source_sha256':test_sha,'driver_sha256':bound[script]})
with (output/'run.log').open('x') as log:
    code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
text=(output/'run.log').read_text()
passed=code==0 and '1 passed; 0 failed' in text and 'test '+name+' ... ok' in text
closure(directory,prep);need(all(sha(p)==h for p,h in bound.items()),'Changed allocation input/source')
write(output/'report.json',{'kind':'attention64-compact-allocations-v1','status':'passed' if passed else 'failed',
    'exit_code':code,'build_sha256':build_sha,'binary_sha256':binary_sha,'test_source_sha256':test_sha,
    'driver_sha256':bound[script],'command':command,'run_log_sha256':sha(output/'run.log'),
    'asserted_warmed_decode_intervals':8,'asserted_allocations_per_interval':0 if passed else None,
    'intervals':'expanded/compact × unpacked/phase-packed × single/batch4, unchanged test',
    'source_and_input_closure':True,'performance_claim':False})
need(passed,'Allocation test failed; preserve output')
print(json.dumps({'status':'passed','report_sha256':sha(output/'report.json')}))
