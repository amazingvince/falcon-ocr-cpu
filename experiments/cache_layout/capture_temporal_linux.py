#!/usr/bin/env python3
"""Build/test the exact frozen native candidate archive under Linux, CPU-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
NATIVE=ROOT/"artifacts/diagnostics/prefix-temporal-candidate-windows-v1"
TARGET=Path("/home/amazi/falcon-ocr-rust-reference/rust-target")
PINNED={
    "reference/prefix-temporal-candidate-windows-v1.json":"d5cda80c9eee15fab3347484e38fdf7c391f6c39279763d586b33004f20b0e93",
    "artifacts/diagnostics/prefix-temporal-candidate-windows-v1/build.json":"a1d075c11cc520d50543e33499c2461fa4a98af54ec51fa5b37440cb879a8cb6",
    "artifacts/diagnostics/prefix-temporal-candidate-windows-v1/execution.json":"c2fc05624f59ef8b777df3d0771a754a08d5863f0f8640ceca276ee2649a3949",
    "artifacts/diagnostics/prefix-temporal-candidate-windows-v1/source.zip":"3c56404aa7a996b0e29b93916a71d50b75cfa033dc33ca8bc12480e51a574176",
    "scripts/build_linux.sh":"25203d4275f25a786bc91531f36d1ebe25ae77736d0dd59b7af1832b0f759e76",
}

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()

def write(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def archive_members(archive,build):
    """Read and hash the same bytes subsequently extracted; exact member inventory."""
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(len(names)==len(set(names)),"Duplicate archive member")
        result={}
        for name in names:
            path=PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,"Invalid source member")
            expected=build["original_source_sha256"].get(name.removeprefix("capture/")) if name.startswith("capture/") else build["isolated_source_sha256"].get(name)
            require(expected is not None,"Unlisted source member: "+name)
            raw=z.read(name)
            require(hashlib.sha256(raw).hexdigest()==expected,"Source member bytes changed: "+name)
            result[name]=raw
        require({n for n in names if not n.startswith("capture/")}==set(build["isolated_source_sha256"]),"Candidate source inventory differs")
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(sys.platform=="linux","This wrapper requires Linux/WSL")
    require(TARGET.is_dir(),"Existing Linux dependency cache is required")
    inputs={ROOT/n:h for n,h in PINNED.items()}
    for path,h in inputs.items():require(sha(path)==h,"Pinned native/build source changed: "+str(path))
    native_build=json.loads((NATIVE/"build.json").read_bytes())
    native_execution=json.loads((NATIVE/"execution.json").read_bytes())
    native_receipt=json.loads((ROOT/"reference/prefix-temporal-candidate-windows-v1.json").read_bytes())
    require(native_execution["status"]=="focused_storage_attention_tests_passed" and native_execution["tests_passed"]==8 and native_execution["source_closure_unchanged"],"Native prerequisite failed")
    require(native_build["isolated_source_sha256"]==native_receipt["isolated_source_sha256"],"Native source identities disagree")
    inputs.update({ROOT/n:h for n,h in native_receipt["evidence_sha256"].items()})
    inputs[Path(__file__).resolve()]=sha(Path(__file__).resolve())
    test_source=Path(__file__).with_name("test_temporal_linux.py")
    inputs[test_source]=sha(test_source)
    for path,h in inputs.items():require(sha(path)==h,"Native/control input changed: "+str(path))
    members=archive_members(NATIVE/"source.zip",native_build)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    project=output/"project";project.mkdir()
    for name,raw in members.items():
        if name.startswith("capture/"):continue
        path=project/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    shutil.copyfile(NATIVE/"source.zip",output/"native-source.zip")
    captures=[Path(__file__).resolve(),test_source,ROOT/"scripts/build_linux.sh"]
    with zipfile.ZipFile(output/"linux-capture-source.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in captures:z.write(path,path.relative_to(ROOT).as_posix())
    source_sha=native_build["isolated_source_sha256"]
    archives={name:sha(output/name) for name in ["native-source.zip","linux-capture-source.zip"]}
    def unchanged():
        require(all(sha(path)==h for path,h in inputs.items()),"Native evidence/control source changed")
        require(all(sha(project/n)==h for n,h in source_sha.items()),"Candidate source changed")
        require({p.relative_to(project).as_posix() for p in project.rglob('*') if p.is_file()}==set(source_sha),"Candidate source inventory changed")
        require(all(sha(output/n)==h for n,h in archives.items()),"Preserved source archive changed")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",CARGO_TARGET_DIR=str(TARGET),CARGO_BUILD_JOBS="2")
    env["PATH"]="/home/amazi/.cargo/bin:"+env.get("PATH","")
    command=["bash",str(ROOT/"scripts/build_linux.sh"),"test","--offline","--locked","--release","--no-run","--lib","--jobs","2","--message-format=json-render-diagnostics","--manifest-path",str(project/"Cargo.toml")]
    whitelist=["CUDA_VISIBLE_DEVICES","CARGO_TARGET_DIR","CARGO_BUILD_JOBS","RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS","CARGO_BUILD_TARGET","CARGO_BUILD_RUSTFLAGS","CC","CXX","CFLAGS","CXXFLAGS","LDFLAGS","ASM_NASM","CMAKE_GENERATOR","FOCR_TOOL_DIR"]
    rustc=subprocess.check_output(["rustc","-Vv"],text=True,env=env)
    require("release: 1.92.0\n" in rustc,"Rust toolchain differs from native")
    build={"schema":1,"status":"prepared","scope":__doc__,"platform":platform.platform(),"python":sys.version,
        "command":command,"rustc_version":rustc,"cargo_version":subprocess.check_output(["cargo","-V"],text=True,env=env),
        "environment":{k:env[k] for k in whitelist if k in env},"input_sha256":{str(p):h for p,h in inputs.items()},
        "isolated_source_sha256":source_sha,"archives_sha256":archives,"native_build_sha256":sha(NATIVE/"build.json"),
        "native_execution_sha256":sha(NATIVE/"execution.json"),"runtime_threads":4,"model_inference":False,"timing":False}
    unchanged();write(output/"build-start.json",build)
    with (output/"build.log").open("x",encoding="utf-8") as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    require(code==0,"Linux build failed; build.log preserved")
    binaries=set()
    for line in (output/"build.log").read_text(encoding="utf-8").splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if row.get("reason")=="compiler-artifact" and row.get("executable") and row.get("profile",{}).get("test"):binaries.add(row["executable"])
    require(len(binaries)==1,"Ambiguous Cargo-emitted test executable")
    emitted=binaries.pop();binary=output/"temporal-candidate-tests";shutil.copyfile(emitted,binary);binary.chmod(0o755)
    unchanged();build.update(status="built",cargo_emitted_executable=emitted,binary=str(binary),binary_sha256=sha(binary),build_log_sha256=sha(output/"build.log"));write(output/"build.json",build)
    listing=subprocess.check_output([str(binary),"temporal_candidate","--list"],cwd=ROOT,env=env,text=True)
    (output/"test-list.txt").write_text(listing,encoding="utf-8",newline="\n")
    selected=[line[:-6] for line in listing.splitlines() if line.endswith(": test")]
    require(selected==native_execution["tests"] and len(selected)==8,"Selected tests differ from native run")
    env["FOCR_TEMPORAL_MEMORY_REPORT"]=str(output/"owned-memory.json")
    run_command=[str(binary),"temporal_candidate","--test-threads=1","--nocapture"]
    with (output/"test.log").open("x",encoding="utf-8") as log:
        code=subprocess.run(run_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    require(code==0,"Linux focused tests failed; test.log preserved")
    log_text=(output/"test.log").read_text(encoding="utf-8")
    require("8 passed; 0 failed; 0 ignored; 0 measured" in log_text,"Missing full test completion")
    require(all("test "+name+" ... ok" in log_text for name in selected),"A selected test did not execute")
    memory=json.loads((output/"owned-memory.json").read_bytes())
    native_memory=json.loads((NATIVE/"owned-memory.json").read_bytes())
    require({k:v for k,v in memory.items() if k!="supported_test_backends"}=={k:v for k,v in native_memory.items() if k!="supported_test_backends"},"Memory/storage accounting differs from native")
    unchanged();require(sha(binary)==build["binary_sha256"],"Linux executable changed")
    write(output/"execution.json",{"schema":1,"status":"same_candidate_linux_focused_tests_passed","command":run_command,
        "tests":selected,"tests_passed":8,"same_source_as_native":True,"same_test_inventory_as_native":True,
        "source_closure_unchanged":True,"input_closure_unchanged":True,"build_sha256":sha(output/"build.json"),
        "binary_sha256":sha(binary),"test_log_sha256":sha(output/"test.log"),"test_list_sha256":sha(output/"test-list.txt"),
        "memory_sha256":sha(output/"owned-memory.json"),"backends":memory["supported_test_backends"],"cuda_visible_devices":env["CUDA_VISIBLE_DEVICES"],
        "model_inference":False,"timing":False,"limits":"Operator/storage tests only. Vec capacity payload is not allocator/process heap peak or RSS. No model or performance qualification."})
    print(json.dumps({"status":"same_candidate_linux_focused_tests_passed","output":str(output),"tests":8,"backends":memory["supported_test_backends"]}))

if __name__=="__main__":main()
