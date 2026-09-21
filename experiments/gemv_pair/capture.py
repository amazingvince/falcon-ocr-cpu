#!/usr/bin/env python3
"""Copy/build/operator phases; no full-model inference or benchmark execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from patch import ROOT, KERNELS_SHA, make_patch, BASELINE_BUILD_SHA, BASELINE_ARCHIVE_SHA, TEST_NAMES, INPUT_PINS

HERE=Path(__file__).resolve().parent
CONTROL=ROOT/'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
CONTROL_BUILD=BASELINE_BUILD_SHA
CONTROL_ARCHIVE=BASELINE_ARCHIVE_SHA
FILTER='kernels::gemv_pair_candidate::tests::'
SOURCES=['candidate.rs','tests.rs','patch.py','capture.py','test_source.py','README.md']
OPERATOR_FIXTURES=INPUT_PINS
ENV_KEYS=['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_BUILD_RUSTFLAGS',
          'CARGO_TARGET_DIR','CC','CXX','CFLAGS','CXXFLAGS','LDFLAGS','ASM_NASM','CMAKE_GENERATOR','FOCR_TOOL_DIR']

def need(ok,message):
    if not ok:raise ValueError(message)
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def read(path,expected):
    raw=path.read_bytes();need(digest(raw)==expected,f'Changed {path}');return json.loads(raw)
def local(path):
    path=path.resolve();need(path.is_relative_to(ROOT) and path!=ROOT,'Output must be inside workspace');return path
def source_map():return {p.relative_to(ROOT).as_posix():sha(p) for p in [HERE/n for n in SOURCES]+[ROOT/'tests/decode_allocations.rs']}

def operator_fixtures():
    for name,h in OPERATOR_FIXTURES.items():need(sha(ROOT/name)==h,'Operator fixture changed: '+name)
def closure(directory,record):
    need(source_map()==record['experiment_source_sha256'],'Experiment source changed')
    need(sha(CONTROL/'build.json')==CONTROL_BUILD and sha(CONTROL/'source.zip')==CONTROL_ARCHIVE,'Control evidence changed')
    for name,h in record['project_source_sha256'].items():need(sha(directory/'project'/name)==h,f'Copied source changed: {name}')
    need(sha(directory/'source.zip')==record['source_archive_sha256'],'Candidate archive changed')

def prepare(args):
    directory=local(args.output);need(not directory.exists(),'Preserve existing output')
    control=read(CONTROL/'build.json',CONTROL_BUILD)
    need(sha(CONTROL/'source.zip')==CONTROL_ARCHIVE,'Wrong frozen source archive')
    sources=source_map()
    with zipfile.ZipFile(CONTROL/'source.zip') as z:
        need(len(z.namelist())==len(set(z.namelist())),'Duplicate archive entries')
        original={n:z.read(n) for n in z.namelist()}
    need({n:digest(v) for n,v in original.items()}==control['source_sha256'],'Archive/source manifest disagreement')
    need(digest(original['src/kernels.rs'])==KERNELS_SHA,'Wrong original kernels')
    patched,_,function=make_patch(original['src/kernels.rs'].decode().replace('\r\n','\n'),
        (HERE/'candidate.rs').read_text(),(HERE/'tests.rs').read_text())
    copied=dict(original);copied['src/kernels.rs']=patched.encode()
    need([n for n in original if copied[n]!=original[n]]==['src/kernels.rs'],'Unexpected code patch')
    fixtures={p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted((ROOT/'tests/fixtures').glob('*.json'))}
    need(fixtures,'Lib compilation fixture inventory is empty')
    fixtures['tests/decode_allocations.rs']=(ROOT/'tests/decode_allocations.rs').read_bytes()
    copied.update(fixtures)
    directory.mkdir(parents=True);project=directory/'project';project.mkdir()
    for name,data in copied.items():
        target=project/name;need(target.resolve().is_relative_to(project),'Unsafe archive path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    with zipfile.ZipFile(directory/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in copied.items():z.writestr(name,data)
    record={'kind':'gemv-pair-preparation-v1','status':'prepared_not_built',
        'control_build_sha256':CONTROL_BUILD,'control_archive_sha256':CONTROL_ARCHIVE,
        'experiment_source_sha256':sources,'project_source_sha256':{n:digest(v) for n,v in copied.items()},
        'source_archive_sha256':sha(directory/'source.zip'),'sole_changed_control_source':'src/kernels.rs',
        'additional_test_fixture_sha256':{n:digest(v) for n,v in fixtures.items()},
        'original_linear_text_sha256':digest(function.encode()),
        'test_filter':FILTER,'selected_test_count':len(TEST_NAMES),'selected_tests':list(TEST_NAMES),
        'operator_fixture_sha256':OPERATOR_FIXTURES,'full_model_execution_allowed':False}
    closure(directory,record);write(directory/'preparation.json',record)
    print(json.dumps({'status':record['status'],'preparation_sha256':sha(directory/'preparation.json')}))

def load(args):
    directory=local(args.prepared);record=read(directory/'preparation.json',args.preparation_sha256)
    need(record['kind']=='gemv-pair-preparation-v1' and record['test_filter']==FILTER,'Wrong preparation')
    need(record['selected_tests']==list(TEST_NAMES) and record['selected_test_count']==len(TEST_NAMES),'Test inventory changed')
    need(record['operator_fixture_sha256']==OPERATOR_FIXTURES,'Operator fixture contract changed')
    closure(directory,record);return directory,record

def build(args):
    need(sys.platform=='win32','Native Windows build required')
    directory,record=load(args);project=directory/'project';target=args.target_dir.resolve()
    need(not target.exists() and target.drive.upper()=='D:','Require fresh unique D: target')
    need(not (directory/'build-start.json').exists(),'Build already attempted')
    env=dict(os.environ,CARGO_TARGET_DIR=str(target))
    control=read(CONTROL/'build.json',CONTROL_BUILD)
    expected={k:v for k,v in control['environment_overrides'].items() if k!='CARGO_TARGET_DIR'}
    actual={k:env[k] for k in ENV_KEYS if k in env and k!='CARGO_TARGET_DIR'}
    need(actual==expected,'Captured build environment differs from frozen control')
    # PATH discovery keeps the archived wrapper from downloading fallback tools.
    nasm=shutil.which('nasm') or str(ROOT/'artifacts/tools/nasm-2.16.03/nasm.exe')
    cmake=shutil.which('cmake') or str(ROOT/'artifacts/tools/cmake-3.31.10-windows-x86_64/bin/cmake.exe')
    need(Path(nasm).is_file() and Path(cmake).is_file(),'Installed native tools required')
    env['PATH']=str(Path(nasm).parent)+os.pathsep+str(Path(cmake).parent)+os.pathsep+env['PATH']
    tools={str(Path(p).resolve()):sha(Path(p)) for p in [nasm,cmake]}
    need(subprocess.check_output(['rustc','-Vv'],text=True,env=env)==control['rustc_version'],'Rust toolchain changed')
    need(subprocess.check_output(['cargo','-V'],text=True,env=env)==control['cargo_version'],'Cargo changed')
    start={'preparation_sha256':args.preparation_sha256,'target_directory':str(target),'tools_sha256':tools,
           'platform':platform.platform(),'python':sys.version,'PATH':env['PATH'],
           'environment_overrides':{k:env[k] for k in ENV_KEYS if k in env}}
    write(directory/'build-start.json',start)
    # Unchanged generic capture yields the expected parent benchmark schema.
    command=[sys.executable,str(project/'scripts/capture_rust_build.py'),'--example','ocr_bench',
             '--output',str(directory/'benchmark-build'),'--cargo-target-dir',str(target),'--jobs','2']
    with (directory/'benchmark-capture.log').open('x') as log:
        code=subprocess.run(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    need(code==0,'Benchmark binary build failed; inspect preserved log')
    benchmark=json.loads((directory/'benchmark-build/build.json').read_bytes())
    need(benchmark['status']=='complete' and benchmark['source_unchanged_during_build'],'Capture rejected build')
    need(benchmark['source_sha256']=={n:record['project_source_sha256'][n] for n in control['source_sha256']},'Benchmark source closure differs')
    need(benchmark['command'][6:]==control['command'][6:],'Cargo build arguments differ')
    benchmark_log=directory/'benchmark-build/build.log'
    benchmark_artifacts=[]
    for line in benchmark_log.read_text(encoding='utf-8').splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if (row.get('reason')=='compiler-artifact' and row.get('executable')
                and row['target']['kind']==['example'] and row['target']['name']=='ocr_bench'):
            benchmark_artifacts.append(row)
    need(len(benchmark_artifacts)==1,'Expected exact unique benchmark Cargo artifact')
    benchmark_artifact=benchmark_artifacts[0]
    benchmark_emitted=Path(benchmark_artifact['executable']).resolve()
    need(benchmark_artifact.get('fresh') is False,'Benchmark reused a prior Cargo artifact')
    need(Path(benchmark_artifact['manifest_path']).resolve()==project/'Cargo.toml','Wrong benchmark Cargo project')
    need(Path(benchmark_artifact['target']['src_path']).resolve()==project/'examples/ocr_bench.rs','Wrong benchmark source')
    need(benchmark_emitted.is_relative_to(target),'Wrong benchmark target directory')
    need(Path(benchmark['cargo_emitted_executable']).resolve()==benchmark_emitted,'Benchmark capture selected different artifact')
    need(sha(benchmark_emitted)==benchmark['binary_sha256']==sha(directory/'benchmark-build'/benchmark['binary']),
         'Benchmark emitted/preserved bytes differ')
    # CLI and lib test binaries are separate targets in this same isolated project.
    wrapper=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(project/'scripts/build_windows.ps1')]
    executables={}
    for label,flags,kind,name in [('cli',['build','--locked','--release','--bin','falcon-ocr'],['bin'],'falcon-ocr'),
        ('operators',['test','--locked','--release','--no-run','--lib'],['lib'],'falcon_ocr'),
        ('allocations',['test','--locked','--release','--no-run','--test','decode_allocations'],['test'],'decode_allocations')]:
        cmd=wrapper+flags+['--jobs','2','--message-format=json-render-diagnostics']
        write(directory/(label+'-invocation.json'),{'command':cmd,'cwd':str(project)})
        logpath=directory/(label+'-build.log')
        with logpath.open('x') as log:code=subprocess.run(cmd,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        need(code==0,f'{label} build failed; preserved log')
        found=[]
        for line in logpath.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('reason')=='compiler-artifact' and row.get('executable') and row['target']['kind']==kind and row['target']['name']==name:found.append(row)
        need(len(found)==1,'Expected exact unique Cargo artifact')
        row=found[0];emitted=Path(row['executable']).resolve()
        need(row.get('fresh') is False and Path(row['manifest_path']).resolve()==project/'Cargo.toml','Wrong/fresh reused project artifact')
        need(Path(row['target']['src_path']).resolve()==project/({'cli':'src/main.rs','operators':'src/lib.rs','allocations':'tests/decode_allocations.rs'}[label]),'Wrong Cargo source')
        need(emitted.is_relative_to(target),'Wrong target directory')
        dest=directory/(label+'.exe');shutil.copyfile(emitted,dest);need(sha(dest)==sha(emitted),'Binary copy changed')
        executables[label]={'path':dest.name,'sha256':sha(dest),'cargo_artifact':row,
                           'log_sha256':sha(logpath),'invocation_sha256':sha(directory/(label+'-invocation.json'))}
    closure(directory,record);need(all(sha(Path(p))==h for p,h in tools.items()),'Native tools changed')
    result={**start,'status':'built_not_executed','kind':'gemv-pair-build-v1','executables':executables,
        'benchmark_build_sha256':sha(directory/'benchmark-build/build.json'),
        'benchmark_cargo_artifact':benchmark_artifact,'benchmark_build_log_sha256':sha(benchmark_log),
        'benchmark_capture_log_sha256':sha(directory/'benchmark-capture.log'),
        'build_start_sha256':sha(directory/'build-start.json')}
    write(directory/'build.json',result);print(json.dumps({'status':result['status'],'build_sha256':sha(directory/'build.json')}))

def operators(args):
    directory,record=load(args);build=read(directory/'build.json',args.build_sha256)
    need(build['preparation_sha256']==args.preparation_sha256 and build['status']=='built_not_executed','Wrong build')
    exe=directory/build['executables']['operators']['path'];expected=build['executables']['operators']['sha256']
    need(sha(exe)==expected,'Test binary changed')
    need(not (directory/'operator-start.json').exists(),'Operator run already attempted')
    operator_fixtures()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',FOCR_GEMV_PAIR_ROOT=str(ROOT))
    listing=subprocess.check_output([str(exe),FILTER,'--list'],env=env,text=True)
    names=[s[:-6] for s in listing.splitlines() if s.endswith(': test')]
    need(names==sorted(FILTER+n for n in TEST_NAMES),'Wrong operator test inventory')
    # Keep each test's standard status line intact; successful real-fixture
    # diagnostics are printed afterwards rather than interleaved with it.
    cmd=[str(exe),FILTER,'--show-output','--test-threads','1']
    write(directory/'operator-start.json',{'command':cmd,'binary_sha256':expected,'tests':names,
        'operator_fixture_sha256':OPERATOR_FIXTURES,'platform':platform.platform()})
    with (directory/'operator.log').open('x') as log:code=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    text=(directory/'operator.log').read_text()
    passed=code==0 and f'{len(TEST_NAMES)} passed; 0 failed' in text and all(f'test {n} ... ok' in text for n in names)
    real_cases=[json.loads(line.partition('GEMV_PAIR_REAL_CASE ')[2]) for line in text.splitlines()
                if 'GEMV_PAIR_REAL_CASE ' in line]
    expected_cases=[
        ('cpu_state.layer.7.attention_norm[112]','layers.7.attention.wqkv.weight',768,2048),
        ('cpu_state.layer.7.attention[112]','layers.7.attention.wo.weight',1024,768),
        ('cpu_state.layer.7.ffn_norm[112]','layers.7.feed_forward.w13.weight',768,4608),
        ('cpu_state.layer.7.gate[112]','layers.7.feed_forward.w2.weight',2304,768),
        ('decode.1.layer.21.hidden+original_final_rms_norm','output.weight',768,65536),
    ]
    real_cases_pass=(len(real_cases)==len(expected_cases) and all(
        (c['input'],c['weight'],c['input_width'],c['output_width'])==expected_case
        and c['compared_outputs']==expected_case[3] and c['bit_mismatches']==0
        and c['original_output_raw_sha256']==c['candidate_output_raw_sha256']
        for c,expected_case in zip(real_cases,expected_cases)))
    passed=passed and real_cases_pass
    operator_fixtures()
    closure(directory,record);need(sha(exe)==expected,'Test binary changed during execution')
    write(directory/'operators.json',{'status':'passed' if passed else 'failed','exit_code':code,
        'preparation_sha256':args.preparation_sha256,'build_sha256':args.build_sha256,
        'binary_sha256':expected,'tests':names,'log_sha256':sha(directory/'operator.log'),
        'source_closure':True,'operator_fixture_sha256':OPERATOR_FIXTURES,
        'real_operand_operator_tests':True,'real_cases':real_cases,
        'real_cases_complete_and_exact':real_cases_pass,
        'full_model_or_benchmark_executed':False})
    need(passed,'Operator tests failed; do not run model')
    print(json.dumps({'status':'passed','report_sha256':sha(directory/'operators.json')}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-authorized',action='store_true',required=True)
    sub=p.add_subparsers(dest='phase',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',type=Path,required=True)
    for phase in ['build','operators']:
        q=sub.add_parser(phase);q.add_argument('--prepared',type=Path,required=True);q.add_argument('--preparation-sha256',required=True)
        q.add_argument('--target-dir' if phase=='build' else '--build-sha256',required=True,**({'type':Path} if phase=='build' else {}))
    a=p.parse_args()
    try:{'prepare':prepare,'build':build,'operators':operators}[a.phase](a)
    except Exception as e:
        directory=local(a.output if a.phase=='prepare' else a.prepared)
        if directory.exists() and not (directory/(a.phase+'-failed.json')).exists():
            write(directory/(a.phase+'-failed.json'),{'status':'failed','error':str(e),'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':main()
