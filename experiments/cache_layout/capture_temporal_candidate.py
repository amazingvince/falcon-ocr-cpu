#!/usr/bin/env python3
"""Copy-only prefix temporal-K storage diagnostic: focused tests, no model inference/timing."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
KERNEL_SHA="19f1f18e164fffcab63dd1d747aae76f3a4a75dfcc15edebacd46b5dc32f7f16"
PINS={
    "reference/prefix-temporal-dedup-design-v1.json":"12293a42533d9891ed9c3dd80e8cfbac0b28bda1a0ae1d5ddd20441ebea070fc",
    "reference/prefix-temporal-dedup-design-v1.md":"69d8a554838a4dfd7daa8fbb9eb7971c3cc99e23927bdd29ef39d02e8bfde181",
    "reference/prefix-temporal-sharing-byte-audit-v2.json":"a63f7afa98369db4ce8dc0640fba41f910620997b0fd62ef74c54753411c44c0",
}

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def write(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def replace_once(text,old,new):
    require(text.count(old)==1,"Patch anchor missing/ambiguous: "+old[:100])
    return text.replace(old,new)

def attention_adapter(source):
    start=source.index("pub fn attention_compact_with_simd(")
    end=source.index("\n#[allow(clippy::too_many_arguments)]\nfn attention_gemm_compact(",start)
    original=source[start:end]
    result=replace_once(original,"pub fn attention_compact_with_simd(","pub(crate) fn attention_temporal_candidate_with_simd(")
    result=replace_once(result,"    prefix_k: &[f32],","    temporal_k: &[f32],\n    spatial_k: &[f32],")
    result=replace_once(result,") {\n    assert!(", ") {\n    assert_eq!((n_heads,n_kv_heads,head_dim),(16,8,64),\"temporal candidate fixed dimensions\");\n    assert_eq!(query_len,1,\"temporal candidate supports only one query\");\n    assert!(")
    result=replace_once(result,"        prefix_k.len(),\n        elements(prefix_len, query_width),\n        \"compact attention prefix K shape\"",
        "        temporal_k.len(),\n        elements(prefix_len, 256),\n        \"temporal attention prefix K shape\"")
    result=replace_once(result,"    assert_eq!(\n        generated_k.len(),", "    assert_eq!(spatial_k.len(),elements(prefix_len,512),\"spatial attention prefix K shape\");\n    assert_eq!(\n        generated_k.len(),")
    gemm_start=result.index("    if query_len >= 4 && selected != Simd::Scalar {")
    gemm_end=result.index("    let dot = dot_kernel(selected);",gemm_start)
    result=result[:gemm_start]+result[gemm_end:]
    result=replace_once(result,"            let mut logits = [0.0_f32; TILE];", "            let mut logits = [0.0_f32; TILE];\n            let mut reconstructed = [0.0_f32; 64];")
    result=replace_once(result,"                        let begin = key * query_width + head * head_dim;\n                        &prefix_k[begin..begin + head_dim]",
        "                        let temporal = (key * 8 + kv_head) * 32;\n                        let spatial = (key * 16 + head) * 32;\n                        reconstructed[..32].copy_from_slice(&temporal_k[temporal..temporal+32]);\n                        reconstructed[32..].copy_from_slice(&spatial_k[spatial..spatial+32]);\n                        &reconstructed[..]")
    require("prefix_k" not in result,"Unpatched prefix reference")
    return "\n// Diagnostic-only adapter: compact arithmetic copied verbatim except K loading.\n#[allow(clippy::too_many_arguments)]\n"+result, hashlib.sha256(original.encode()).hexdigest()

def patch_copy(project):
    before={str(p.relative_to(project)).replace("\\","/"):p.read_text(encoding="utf-8") for p in (project/"src").glob("*.rs")}
    kernel=before["src/kernels.rs"]
    adapter,attention_sha=attention_adapter(kernel)
    (project/"src/kernels.rs").write_text(kernel+adapter,encoding="utf-8",newline="\n")
    config=replace_once(before["src/config.rs"],"    Expanded,\n    Compact,\n}","    Expanded,\n    Compact,\n    /// Diagnostic-only copied-project prefix temporal sharing.\n    TemporalCandidate,\n}")
    (project/"src/config.rs").write_text(config,encoding="utf-8",newline="\n")
    (project/"src/lib.rs").write_text(before["src/lib.rs"]+"\nmod temporal_candidate;\n",encoding="utf-8",newline="\n")
    model=before["src/model.rs"]
    model=replace_once(model,"enum LayerCache {\n", "enum LayerCache {\n    Temporal(crate::temporal_candidate::TemporalCache),\n")
    begin=model.index("impl LayerCache {");end=model.index("\n/// Store one original GQA head",begin)
    block=model[begin:end]
    block=replace_once(block,"fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) {\n        match self {",
        "fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) -> Result<()> {\n        match self {\n            Self::Temporal(cache) => cache.append(k,v,offset)?,")
    block=replace_once(block,"        }\n    }\n\n    #[allow", "        }\n        Ok(())\n    }\n\n    #[allow")
    block=replace_once(block,"        q: &[f32],\n", "        q: &[f32],\n        current_expanded_k: &[f32],\n")
    block=replace_once(block,"    ) {\n        match self {", "    ) -> Result<()> {\n        match self {\n            Self::Temporal(cache) => cache.attention(q,current_expanded_k,rows,total_len,offset,image_start,image_end,sinks,output,simd)?,")
    require(block.endswith("        }\n    }\n}\n"),"Unexpected LayerCache block ending")
    block=block[:-len("        }\n    }\n}\n")]+"        }\n        Ok(())\n    }\n}\n"
    model=model[:begin]+block+model[end:]
    model=replace_once(model,"cache.append(&work.k, &work.v, offset, c);","cache.append(&work.k, &work.v, offset, c)?;")
    model=replace_once(model,"cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c);","cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c)?;")
    model=replace_once(model,"            cache.attention(\n                &work.q,", "            cache.attention(\n                &work.q,\n                &work.k,")
    model=replace_once(model,"                cache.attention(\n                    &work.q[range.clone()],", "                cache.attention(\n                    &work.q[range.clone()],\n                    &work.k[range.clone()],")
    for marker in ["            cache.attention(","                cache.attention("]:
        # Exact close of each call; adding Result propagation does not change arithmetic.
        begin=model.index(marker);end=model.index("\n            }" if marker.startswith("                ") else "\n            if trace.enabled()",begin)
        call=model[begin:end]
        require(call.endswith(");"),"Unexpected attention call close")
        model=model[:begin]+call[:-2]+");".replace(");", ")?;")+model[end:]
    model=replace_once(model,"            layers.push(match cache_layout {", "            layers.push(match cache_layout {\n                CacheLayout::TemporalCandidate => LayerCache::Temporal(crate::temporal_candidate::TemporalCache::new(prefix_len,capacity,c.n_heads,c.n_kv_heads,c.head_dim)?),")
    model+='\n#[cfg(test)]\n#[path = "temporal_model_tests.rs"]\nmod temporal_candidate_tests;\n'
    (project/"src/model.rs").write_text(model,encoding="utf-8",newline="\n")
    for name in ["temporal_candidate.rs","temporal_model_tests.rs"]: shutil.copyfile(HERE/name,project/"src"/name)
    diff=[]
    for name in ["src/config.rs","src/lib.rs","src/model.rs","src/kernels.rs"]:
        diff.extend(difflib.unified_diff(before[name].splitlines(True),(project/name).read_text(encoding="utf-8").splitlines(True),fromfile="live/"+name,tofile="isolated/"+name))
    return "".join(diff),attention_sha

def source_paths():
    return [ROOT/n for n in ["Cargo.toml","Cargo.lock","rust-toolchain.toml"]]+sorted((ROOT/"src").glob("*.rs"))+sorted((ROOT/"examples/support").glob("*.rs"))+sorted((ROOT/"tests/fixtures").glob("*.json"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    require(sys.platform=="win32","This capture uses the native Windows build wrapper")
    require(sha(ROOT/"src/kernels.rs")==KERNEL_SHA,"Original arithmetic source changed")
    require(all(sha(ROOT/n)==h for n,h in PINS.items()),"Frozen design/audit prerequisite changed")
    paths=source_paths()
    controls=[Path(__file__).resolve(),HERE/"temporal_candidate.rs",HERE/"temporal_model_tests.rs",HERE/"test_temporal_candidate.py",ROOT/"scripts/build_windows.ps1"]+[ROOT/n for n in PINS]
    original={p.relative_to(ROOT).as_posix():sha(p) for p in paths+controls}
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    project=output/"project";project.mkdir()
    for path in paths:
        dest=project/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    diff,attention_sha=patch_copy(project)
    (output/"candidate.patch").write_text(diff,encoding="utf-8",newline="\n")
    copied={p.relative_to(project).as_posix():sha(p) for p in sorted(project.rglob("*")) if p.is_file()}
    expected_changed={"src/config.rs","src/lib.rs","src/model.rs","src/kernels.rs","src/temporal_candidate.rs","src/temporal_model_tests.rs"}
    require({n for n,h in copied.items() if original.get(n)!=h}==expected_changed,"Unexpected copy patch scope")
    with zipfile.ZipFile(output/"source.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for n in copied: archive.write(project/n,n)
        for p in controls: archive.write(p,"capture/"+p.relative_to(ROOT).as_posix())
    build={"schema":1,"status":"prepared","scope":__doc__,"platform":platform.platform(),"python":sys.version,
        "original_source_sha256":original,"isolated_source_sha256":copied,"source_archive_sha256":sha(output/"source.zip"),
        "patch_sha256":sha(output/"candidate.patch"),"copied_compact_attention_function_sha256":attention_sha,
        "production_kernel_sha256":KERNEL_SHA,"proofs":PINS,"runtime_threads":4,"model_inference":False,"timing":False,
        "limits":"Correctness-first storage adapter only. No full-model/real-output/peak-RSS/performance/GPU numerical qualification."}
    def unchanged():
        require(all(sha(ROOT/n)==h for n,h in original.items()),"Live/capture source changed")
        require(all(sha(project/n)==h for n,h in copied.items()),"Copied source changed")
        require(sha(output/"source.zip")==build["source_archive_sha256"],"Source archive changed")
        require(sha(output/"candidate.patch")==build["patch_sha256"],"Patch changed")
    unchanged();write(output/"preparation.json",build)
    if args.prepare_only:
        print(json.dumps({"status":"prepared_only","project":str(project)}));return
    env=dict(os.environ,CARGO_TARGET_DIR=str(ROOT/"target"))
    command=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts/build_windows.ps1"),
        "test","--offline","--locked","--release","--no-run","--lib","--jobs","2","--message-format=json-render-diagnostics","--manifest-path",str(project/"Cargo.toml")]
    whitelist=["CARGO_TARGET_DIR","RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS","CARGO_BUILD_TARGET","CARGO_BUILD_RUSTFLAGS","CC","CXX","CFLAGS","CXXFLAGS","LDFLAGS","ASM_NASM","CMAKE_GENERATOR","FOCR_TOOL_DIR"]
    build.update(command=command,rustc_version=subprocess.check_output(["rustc","-Vv"],text=True),cargo_version=subprocess.check_output(["cargo","-V"],text=True),environment={k:env[k] for k in whitelist if k in env})
    write(output/"build-start.json",build)
    with (output/"build.log").open("x",encoding="utf-8") as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    require(code==0,"Build failed; preserved build.log")
    binaries=set()
    for line in (output/"build.log").read_text(encoding="utf-8").splitlines():
        try: row=json.loads(line)
        except ValueError: continue
        if row.get("reason")=="compiler-artifact" and row.get("executable") and row.get("profile",{}).get("test"): binaries.add(row["executable"])
    require(len(binaries)==1,"Missing/ambiguous emitted test executable")
    binary=output/"temporal-candidate-tests.exe";shutil.copyfile(binaries.pop(),binary)
    unchanged();build.update(status="built",binary=str(binary),binary_sha256=sha(binary),build_log_sha256=sha(output/"build.log"));write(output/"build.json",build)
    listing=subprocess.check_output([str(binary),"temporal_candidate","--list"],cwd=ROOT,env=env,text=True)
    selected=[line[:-6] for line in listing.splitlines() if line.endswith(": test")]
    require(len(selected)==8 and all("temporal_candidate::tests::" in n or "temporal_candidate_tests::" in n for n in selected),"Focused test inventory differs")
    env["FOCR_TEMPORAL_MEMORY_REPORT"]=str(output/"owned-memory.json")
    run_command=[str(binary),"temporal_candidate","--test-threads=1","--nocapture"]
    with (output/"test.log").open("x",encoding="utf-8") as log:
        code=subprocess.run(run_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    require(code==0,"Focused tests failed; preserved test.log")
    require("8 passed; 0 failed" in (output/"test.log").read_text(encoding="utf-8"),"Focused test completion missing")
    memory=json.loads((output/"owned-memory.json").read_bytes())
    require(memory["measured_candidate_buffer_capacity_bytes"]==memory["candidate_analytic_payload_bytes"],"Owned payload differs")
    unchanged();require(sha(binary)==build["binary_sha256"],"Executable changed")
    write(output/"execution.json",{"schema":1,"status":"focused_storage_attention_tests_passed","command":run_command,
        "tests":selected,"tests_passed":len(selected),"source_closure_unchanged":True,"build_sha256":sha(output/"build.json"),
        "binary_sha256":sha(binary),"test_log_sha256":sha(output/"test.log"),"memory_sha256":sha(output/"owned-memory.json"),
        "model_inference":False,"timing":False,"limits":build["limits"]})
    print(json.dumps({"status":"focused_storage_attention_tests_passed","output":str(output),"tests":len(selected)}))

if __name__=="__main__": main()
