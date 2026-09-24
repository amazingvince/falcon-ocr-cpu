"""Bounded WSL CPU qualification of the unchanged combined Windows source ZIP."""
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[4]
NATIVE=ROOT/'artifacts/diagnostics/attention64-compact-v1'
OUTPUT=ROOT/'artifacts/diagnostics/attention64-compact-linux-v1'
TARGET=Path('/home/amazi/falcon-ocr-rust-reference/attention64-compact-linux-v1-target')
CONTROL_DIR=ROOT/'artifacts/diagnostics/prefix-temporal-model-linux-plan-v1'
FILTER='kernels::attention64_candidate::tests::'
PINS={
 'artifacts/diagnostics/attention64-compact-v1/preparation.json':'1862072c940d9aa2821562f7d1b94e6d7493692b10502aa60dafd2e8ce9458ce',
 'artifacts/diagnostics/attention64-compact-v1/source.zip':'c2f9e8a7385466d1abe8231f506d06c0f193aeb7f50621acc46ee2e6b0a4b43e',
 'artifacts/diagnostics/attention64-compact-v1/operators.json':'52c0e5e261decc2904f535b06fb289455c5caa45be70879a26d4b5e4bd68e9ff',
 'artifacts/diagnostics/attention64-compact-v1/smoke/report.json':'1cf1142f58ed688e45ca9797a2520ea593a09c0e337b0ca23376bdccdc46ea5b',
 'artifacts/diagnostics/attention64-compact-v1/allocations/report.json':'350d80d07bc7bf632cb3419db5410f12fd6e8075a8b1b81055074a84c1dad900',
 'reference/prefix-temporal-model-linux-v1.json':'5272bcdfdc29391a9f11fc5ee05407b0ad2cea3834db3ae920d7862f5a9fae17',
 'artifacts/diagnostics/prefix-temporal-model-linux-plan-v1/expanded-before.json':'dddf246344e0b733bdb910a5b432fbfb0d43c9c69948d1651a6ac82e1ba3df6d',
 'artifacts/reference/smoke-fp32/trace.safetensors':'30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4',
}

def need(ok,message):
    if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path,expected):
    raw=path.read_bytes();need(digest(raw)==expected,'Changed '+str(path));return json.loads(raw)
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(command,label,env):
    write(OUTPUT/(label+'-invocation.json'),{'command':command,'cwd':str(ROOT)})
    with (OUTPUT/(label+'.log')).open('x',encoding='utf-8') as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    need(code==0,label+' failed; log preserved')
    return (OUTPUT/(label+'.log')).read_text(encoding='utf-8')

def main():
    need(sys.platform=='linux' and 'microsoft' in platform.release().lower(),'Expected Linux under WSL')
    need(not TARGET.exists() and not OUTPUT.exists(),'Fresh unique output and target required')
    bound={ROOT/n:h for n,h in PINS.items()}
    for p,h in bound.items():need(sha(p)==h,'Pinned input changed '+str(p))
    script=Path(__file__).resolve();bound[script]=sha(script)
    prep=read(NATIVE/'preparation.json',bound[NATIVE/'preparation.json'])
    summary=read(ROOT/'reference/prefix-temporal-model-linux-v1.json',PINS['reference/prefix-temporal-model-linux-v1.json'])
    need(summary['status']=='bounded_linux_cpu_layout_qualification_passed' and summary['source_and_artifact_closure_rechecked'],'Saved Linux control not accepted')
    control=read(CONTROL_DIR/'expanded-before.json',bound[CONTROL_DIR/'expanded-before.json'])
    need(control['status']=='completed' and control['cache_layout']=='expanded','Wrong saved control')
    need(control['runtime']['threads']==4 and control['runtime']['backend']=='avx2','Wrong control runtime')
    base=CONTROL_DIR.relative_to(ROOT).as_posix()+'/'
    # Recheck the control's actual source, binary, results and saved provenance;
    # older unrelated Windows artifacts remain bound by the accepted summary.
    for n,h in summary['checked_files_sha256'].items():
        tail=n.removeprefix(base)
        if n.startswith(base) and (tail.startswith('control/') or tail in ['control-qualification','expanded-before.json','expanded-before.log',
            'expanded-before-invocation.json','plan.json','build.json','execution.json','comparison.json']):bound[ROOT/n]=h
    original=read(ROOT/'artifacts/builds/ocr-bench-fullpages-v2-windows/build.json',prep['control_build_sha256'])
    bound[ROOT/'artifacts/builds/ocr-bench-fullpages-v2-windows/build.json']=prep['control_build_sha256']
    for n,h in original['source_sha256'].items():
        if n.startswith('src/') or n in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:
            need(summary['checked_files_sha256'].get(base+'control/'+n)==h,'Saved Linux control has different runtime source '+n)
    need(control['weights_sha256']=='3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16','Wrong control model')
    for p,h in bound.items():need(sha(p)==h,'Control provenance changed '+str(p))
    with zipfile.ZipFile(NATIVE/'source.zip') as z:
        names=z.namelist();need(len(names)==len(set(names)),'Duplicate source member')
        members={n:z.read(n) for n in names}
    need({n:digest(v) for n,v in members.items()}==prep['project_source_sha256'],'Source ZIP differs from native candidate')
    OUTPUT.mkdir(parents=True);project=OUTPUT/'project';project.mkdir()
    for name,data in members.items():
        path=PurePosixPath(name);need(not path.is_absolute() and '..' not in path.parts and '\\' not in name,'Unsafe archive path')
        dest=project/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    shutil.copyfile(NATIVE/'source.zip',OUTPUT/'native-source.zip')
    shutil.copyfile(script,OUTPUT/'capture-source.py')
    def closure():
        need(all(sha(p)==h for p,h in bound.items()),'Input/source identity changed')
        need(all(sha(project/n)==h for n,h in prep['project_source_sha256'].items()),'Copied candidate source changed')
        need({p.relative_to(project).as_posix() for p in project.rglob('*') if p.is_file()}==set(members),'Copied inventory changed')
        need(sha(OUTPUT/'native-source.zip')==prep['source_archive_sha256'],'Preserved archive changed')
        need(sha(OUTPUT/'capture-source.py')==bound[script],'Preserved helper changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',CARGO_TARGET_DIR=str(TARGET),CARGO_BUILD_JOBS='2')
    toolpaths=[Path('/home/amazi/.cargo/bin'),ROOT/'artifacts/tools/linux/cmake-3.31.10-linux-x86_64/bin',ROOT/'artifacts/tools/linux/nasm-installed/bin']
    env['PATH']=os.pathsep.join(map(str,toolpaths))+os.pathsep+env.get('PATH','')
    rust=subprocess.check_output(['rustc','-Vv'],text=True,cwd=project,env=env)
    need('release: 1.92.0\n' in rust,'Wrong Rust toolchain')
    tools={name:shutil.which(name,path=env['PATH']) for name in ['cargo','rustc','cc','cmake','nasm']}
    need(all(tools.values()),'Preinstalled tool missing; no installation allowed')
    for p in tools.values():bound[Path(p)]=sha(Path(p))
    whitelist=['CUDA_VISIBLE_DEVICES','CARGO_TARGET_DIR','CARGO_BUILD_JOBS','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS',
        'CARGO_BUILD_TARGET','CARGO_BUILD_RUSTFLAGS','CC','CXX','CFLAGS','CXXFLAGS','LDFLAGS','ASM_NASM','CMAKE_GENERATOR','FOCR_TOOL_DIR']
    start={'kind':'attention64-compact-linux-v1','platform':platform.platform(),'python':sys.version,
        'rustc_version':rust,'cargo_version':subprocess.check_output(['cargo','-V'],text=True,env=env),
        'environment':{k:env[k] for k in whitelist if k in env},'tools':tools,'PATH':env['PATH'],
        'source_sha256':prep['project_source_sha256'],'native_source_archive_sha256':prep['source_archive_sha256'],
        'input_sha256':{str(p):h for p,h in bound.items()},'performance_claim':False}
    closure();write(OUTPUT/'start.json',start)
    binaries={}
    for label,flags,kind,name,src in [
        ('cli',['build','--bin','falcon-ocr'],['bin'],'falcon-ocr','src/main.rs'),
        ('operators',['test','--no-run','--lib'],['lib'],'falcon_ocr','src/lib.rs'),
        ('allocations',['test','--no-run','--test','decode_allocations'],['test'],'decode_allocations','tests/decode_allocations.rs')]:
        command=['cargo']+flags+['--offline','--locked','--release','--jobs','2','--message-format=json-render-diagnostics','--manifest-path',str(project/'Cargo.toml')]
        text=run(command,label+'-build',env);found=[]
        for line in text.splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if r.get('reason')=='compiler-artifact' and r.get('executable') and r['target']['kind']==kind and r['target']['name']==name:found.append(r)
        need(len(found)==1,'Ambiguous emitted executable')
        row=found[0];exe=Path(row['executable']).resolve()
        need(row.get('fresh') is False and Path(row['manifest_path']).resolve()==project/'Cargo.toml','Wrong/reused project artifact')
        need(Path(row['target']['src_path']).resolve()==project/src and exe.is_relative_to(TARGET),'Wrong source/target artifact')
        saved=OUTPUT/label;shutil.copyfile(exe,saved);saved.chmod(0o755)
        need(sha(saved)==sha(exe),'Binary copy mismatch');bound[saved]=sha(saved)
        binaries[label]={'path':str(saved),'sha256':bound[saved],'cargo_artifact':row}
    closure();write(OUTPUT/'build.json',{**start,'status':'built','executables':binaries})
    exe=binaries['operators']['path']
    listing=subprocess.check_output([exe,FILTER,'--list'],text=True,env=env)
    names=[x[:-6] for x in listing.splitlines() if x.endswith(': test')]
    native_tests=json.loads((NATIVE/'operators.json').read_bytes())['tests']
    need(names==native_tests and len(names)==11,'Wrong native/Linux test inventory')
    text=run([exe,FILTER,'--nocapture','--test-threads','1'],'operator-tests',env)
    need('11 passed; 0 failed' in text and all('test '+n+' ... ok' in text for n in names),'Some operators did not execute/pass')
    doctor=json.loads(subprocess.check_output([binaries['cli']['path'],'doctor'],text=True,env=env))
    write(OUTPUT/'doctor.json',doctor);need(doctor['avx2'] and doctor['fma'],'AVX2/FMA unavailable')
    trace=OUTPUT/'smoke.safetensors'
    command=[binaries['cli']['path'],'--model',str(ROOT/'artifacts/model'),'--threads','4','--backend','avx2',
        '--precision','fp32','--batch-size','1','--cache-layout','compact','--weight-layout','unpacked','trace',
        '--fixture',str(ROOT/'artifacts/reference/smoke-fp32/trace.safetensors'),'--output',str(trace),'--max-new-tokens','17']
    run(command,'smoke',env)
    raw=trace.read_bytes();header_len=int.from_bytes(raw[:8],'little');header=json.loads(raw[8:8+header_len]);payload=memoryview(raw)[8+header_len:]
    tensors={}
    for n,t in header.items():
        if n=='__metadata__':continue
        lo,hi=t['data_offsets'];need(t['dtype']=='F32' and 0<=lo<=hi<=len(payload),'Invalid F32 tensor')
        elements=math.prod(t['shape']);need(hi-lo==elements*4,'Bad tensor size')
        tensors[n]={'dtype':'F32-le','elements':elements,'shape':t['shape'],'sha256':digest(payload[lo:hi])}
    expected=control['canonical']['trace']['tensors']
    need(len(tensors)==1904 and set(tensors)==set(expected),'Wrong canonical tensor inventory')
    differences=[n for n in tensors if tensors[n]!=expected[n]]
    write(OUTPUT/'smoke-comparison.json',{'tensor_count':len(tensors),'bitwise_hash_mismatches':differences,
        'trace_sha256':digest(raw),'saved_linux_control_sha256':PINS['artifacts/diagnostics/prefix-temporal-model-linux-plan-v1/expanded-before.json'],
        'tensor_comparison':'Complete name/dtype/shape/element count and SHA256 of raw little-endian F32 bytes.',
        'reference_storage':'Saved complete tensor hashes, not a newly generated control or Windows tensor target.'})
    need(not differences,'Linux smoke tensors differ; stop without further model execution')
    result=json.loads(trace.with_suffix('.json').read_bytes());expected_result=control['canonical']['result']
    need(all(result.get(k)==v for k,v in expected_result.items()),'IDs/text/stops/config differ from Linux control')
    need(result['cache_layout']=='compact' and len(result['token_ids'])==17 and result['finish_reason']=='length' and result['teacher_forced'],'Wrong compact teacher trace')
    exe=binaries['allocations']['path'];name='warm_fp32_decode_has_no_heap_allocations'
    listing=subprocess.check_output([exe,'--list'],text=True,env=env)
    need(listing.splitlines().count(name+': test')==1,'Allocation test missing')
    text=run([exe,name,'--exact','--ignored','--nocapture','--test-threads','1'],'allocation-tests',env)
    need('1 passed; 0 failed' in text and 'test '+name+' ... ok' in text,'Allocation check incomplete')
    closure()
    outputs={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    write(OUTPUT/'report.json',{'kind':'attention64-compact-linux-functional-v1','status':'passed',
        'same_source_archive_as_native':True,'source_and_input_closure':True,'operator_tests':names,
        'tensor_count':1904,'tensor_hash_mismatches':0,'teacher_tokens':17,'ids_text_stop_exact':True,
        'allocation_intervals':8,'asserted_allocations_each':0,'backend_support':doctor,
        'build_sha256':sha(OUTPUT/'build.json'),'output_sha256':outputs,
        'performance_claim':False,'gpu_numerical_qualification':False,
        'limits':['WSL compatibility/functional evidence only; no Linux bare-metal timing.',
            'Exact comparison is against saved same-platform complete tensor hashes; cross-platform tensor equality is not required.',
            'The original ten GPU intermediate gates remain open.']})
    print(json.dumps({'status':'passed','report_sha256':sha(OUTPUT/'report.json'),'output':str(OUTPUT)}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUTPUT.exists() and not (OUTPUT/'failed.json').exists():write(OUTPUT/'failed.json',{'status':'failed','error':str(e),'traceback':traceback.format_exc()})
        raise
