//! Isolated actual-operands matrix comparison. No Model/Runner or GPU work.
//! `baseline_kernels.rs` is copied byte-for-byte from the frozen combined baseline.
mod candidate;
#[allow(dead_code)]
#[path = "baseline_kernels.rs"]
mod kernels;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

const THREADS: usize = 16;
const WARMUPS: usize = 2;
const SAMPLES: usize = 7;
const ITERATIONS: usize = 5;
const FIXTURE: (&str, &str) = ("artifacts/reference/linear-operators.safetensors",
    "a8b9c60e8f5da012efca80fa6955edff814616f6372668229e0aba74e6f3d338");
const SIDECAR: (&str, &str) = ("artifacts/reference/linear-operators.json",
    "2cd585ef48c59c9b821e0eae770b2e9c643c4697d239f4b63cac12508cf0e60f");
const CHECKPOINT: (&str, &str) = ("artifacts/model/model.safetensors",
    "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16");
const CONFIG: (&str, &str) = ("artifacts/model/config.json",
    "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf");
const TRACE: (&str, &str) = ("artifacts/cpu/smoke-trace-sinks-pairwise.safetensors",
    "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309");
const BASELINE_KERNELS_SHA: &str = "ad812805e03b536de0408d6f0dd21d3e944d6d752bc510331c59da44fe80fc5c";

// Separate warm-operation intervals only. Counters remain disabled throughout
// timing. This sees Rust GlobalAlloc requests across all pool threads; it does
// not see arbitrary native allocations or report RSS/allocator usable size.
struct CountingAllocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static FREE_BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNT.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && COUNT.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNT.load(Ordering::Relaxed) {
            FREES.fetch_add(1, Ordering::Relaxed);
            FREE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, size) };
        if !new_ptr.is_null() && COUNT.load(Ordering::Relaxed) {
            REALLOCS.fetch_add(1, Ordering::Relaxed);
            REALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        new_ptr
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct CounterScope;
impl Drop for CounterScope {
    fn drop(&mut self) { COUNT.store(false, Ordering::SeqCst); }
}

fn allocation_interval(f: impl FnOnce() -> Result<()>) -> Result<Value> {
    ensure!(!COUNT.load(Ordering::SeqCst), "nested allocation interval");
    for counter in [&ALLOCS, &ALLOC_BYTES, &REALLOCS, &REALLOC_BYTES, &FREES, &FREE_BYTES] {
        counter.store(0, Ordering::SeqCst);
    }
    COUNT.store(true, Ordering::SeqCst);
    let scope = CounterScope;
    let result = f();
    drop(scope);
    result?;
    Ok(json!({"alloc_calls":ALLOCS.load(Ordering::SeqCst),
        "alloc_requested_bytes":ALLOC_BYTES.load(Ordering::SeqCst),
        "realloc_calls":REALLOCS.load(Ordering::SeqCst),
        "realloc_new_requested_bytes":REALLOC_BYTES.load(Ordering::SeqCst),
        "dealloc_calls":FREES.load(Ordering::SeqCst),
        "dealloc_layout_bytes":FREE_BYTES.load(Ordering::SeqCst),
        "scope":"one warm ordered suite; Rust global allocator only; not retained bytes or RSS"}))
}

#[derive(Parser)]
struct Args {
    /// Repository root containing pinned saved operands; never a model run.
    #[arg(long)]
    root: PathBuf,
    /// Must not exist. All errors after creation preserve failed.json.
    #[arg(long)]
    output: PathBuf,
    /// Explicitly enable fixed 2 warmups / 7 samples / 5 suite iterations.
    #[arg(long)]
    measure: bool,
    /// Also inspect the four saved layer0 prefill matrices, M=144 only.
    #[arg(long)]
    prefill: bool,
}

fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn file_sha(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut block = [0_u8; 65536];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 { break; }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn write_json(path: &Path, value: &Value) -> Result<String> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let expected = sha(&bytes);
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(expected)
}

struct BoundMap { path: PathBuf, expected: &'static str, map: Mmap }
impl BoundMap {
    fn new(root: &Path, pin: (&str, &'static str)) -> Result<Self> {
        let path = root.join(pin.0);
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // Read-only maps remain alive through all borrowed tensor views. Inputs
        // must be immutable during this run; mapped and current-path hashes are
        // checked again at closure to catch ordinary drift/replacement.
        let map = unsafe { MmapOptions::new().map(&file)? };
        ensure!(sha(&map) == pin.1, "wrong pinned input {}", path.display());
        Ok(Self { path, expected: pin.1, map })
    }
    fn check(&self) -> Result<()> {
        ensure!(sha(&self.map) == self.expected && file_sha(&self.path)? == self.expected,
            "input changed: {}", self.path.display());
        Ok(())
    }
}
fn tensor<'a>(file: &SafeTensors<'a>, name: &str, shape: &[usize]) -> Result<&'a [f32]> {
    let view = file.tensor(name).with_context(|| format!("tensor {name}"))?;
    ensure!(view.dtype() == Dtype::F32 && view.shape() == shape, "wrong shape/dtype {name}");
    let values = bytemuck::try_cast_slice::<u8, f32>(view.data())
        .map_err(|error| anyhow::anyhow!("unaligned FP32 tensor {name}: {error}"))?;
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite operand {name}");
    Ok(values)
}

struct Case<'a> {
    name: String, weight_key: String, m: usize, k: usize, n: usize,
    input: Cow<'a, [f32]>, weight: &'a [f32], gpu: Option<&'a [f32]>,
    saved_cpu: Option<&'a [f32]>,
}
fn fixture_case<'a>(file: &SafeTensors<'a>, metadata: &Value, name: &str,
                   layer: usize, op: &str, m: usize, k: usize, n: usize) -> Result<Case<'a>> {
    let suffix = match op { "qkv" => "attention.wqkv", "wo" => "attention.wo",
        "w13" => "feed_forward.w13", "w2" => "feed_forward.w2", _ => unreachable!() };
    let weight_key = format!("weights.layers.{layer}.{suffix}.weight");
    let entries = metadata["fixtures"].as_array().context("missing fixture metadata")?;
    let matches: Vec<_> = entries.iter().filter(|item| item["name"].as_str() == Some(name)).collect();
    ensure!(matches.len() == 1 && matches[0]["weight_key"].as_str() == Some(weight_key.as_str()),
        "wrong/multiple fixture mapping for {name}");
    Ok(Case { name: name.to_owned(), weight_key: weight_key.clone(), m, k, n,
        input: Cow::Borrowed(tensor(file, &format!("{name}.input"), &[m, k])?),
        weight: tensor(file, &weight_key, &[n, k])?,
        gpu: Some(tensor(file, &format!("{name}.expected"), &[m, n])?), saved_cpu: None })
}

fn comparison(reference: &[f32], actual: &[f32]) -> Result<Value> {
    ensure!(reference.len() == actual.len() && !actual.is_empty(), "comparison shape");
    ensure!(reference.iter().chain(actual).all(|x| x.is_finite()), "nonfinite output");
    let mut max = 0_f64; let mut index = 0; let mut squared = 0_f64; let mut bits = 0;
    for (i, (&a, &b)) in reference.iter().zip(actual).enumerate() {
        let error = b as f64 - a as f64;
        if error.abs() > max { max = error.abs(); index = i; }
        squared += error * error;
        bits += usize::from(a.to_bits() != b.to_bits());
    }
    Ok(json!({"elements":actual.len(),"different_bits":bits,"max_abs":max,
        "rms_abs":(squared / actual.len() as f64).sqrt(),"worst_flat_index":index,
        "reference_at_worst":reference[index],"actual_at_worst":actual[index]}))
}

struct Oracle { rows: Vec<usize>, values: Vec<f64>, absolute_products: Vec<f64> }
fn oracle(case: &Case<'_>) -> Oracle {
    let rows = if case.m == 1 { vec![0] } else { vec![0, 112, 143] };
    let mut values = vec![0.; rows.len() * case.n];
    let mut absolute_products = values.clone();
    values.par_chunks_mut(case.n).zip(absolute_products.par_chunks_mut(case.n))
        .enumerate().for_each(|(selected, (out, sums))| {
            let x = &case.input[rows[selected] * case.k..(rows[selected] + 1) * case.k];
            for channel in 0..case.n {
                let w = &case.weight[channel * case.k..(channel + 1) * case.k];
                let mut sum = 0_f64; let mut absolute = 0_f64;
                for (&a, &b) in x.iter().zip(w) {
                    let product = a as f64 * b as f64;
                    sum += product; absolute += product.abs();
                }
                out[channel] = sum; sums[channel] = absolute;
            }
        });
    Oracle { rows, values, absolute_products }
}
fn oracle_error(case: &Case<'_>, oracle: &Oracle, output: &[f32]) -> Value {
    let u32 = 2_f64.powi(-24); let u64 = 2_f64.powi(-53);
    let gamma32 = case.k as f64 * u32 / (1. - case.k as f64 * u32);
    let gamma64 = case.k as f64 * u64 / (1. - case.k as f64 * u64);
    let mut max = 0_f64; let mut index = 0; let mut squared = 0_f64;
    let mut reference_squared = 0_f64; let mut outside = 0;
    for (selected, &row) in oracle.rows.iter().enumerate() {
        for channel in 0..case.n {
            let i = selected * case.n + channel;
            let expected = oracle.values[i];
            let error = output[row * case.n + channel] as f64 - expected;
            if error.abs() > max { max = error.abs(); index = i; }
            squared += error * error; reference_squared += expected * expected;
            // F32 products are exact in F64; only sequential F64 summation rounds.
            // Add its conservative uncertainty to the standard gamma_K FP32-dot
            // envelope. Diagnostic only: no frozen model policy is changed.
            let upper_abs_sum = oracle.absolute_products[i] / (1. - gamma64);
            outside += usize::from(error.abs() > (gamma32 + gamma64) * upper_abs_sum);
        }
    }
    json!({"selected_rows":oracle.rows,"elements":oracle.values.len(),"max_abs":max,
        "rms_abs":(squared / oracle.values.len() as f64).sqrt(),
        "relative_l2":if reference_squared > 0. {Some((squared/reference_squared).sqrt())} else {None},
        "worst_row":oracle.rows[index / case.n],"worst_channel":index % case.n,
        "outside_gamma_k_plus_f64_uncertainty":outside,
        "oracle":"sequential F64 products/sum, not rounded back to FP32",
        "bound_scope":"arithmetic diagnostic, not a full-model tolerance or GPU parity gate"})
}

fn baseline(case: &Case<'_>, out: &mut [f32]) {
    kernels::linear_with_simd(black_box(&case.input), case.m, case.k,
        black_box(case.weight), case.n, black_box(out), kernels::Simd::Avx2);
}
fn suite(cases: &[Case<'_>], prepared: &[candidate::Prepared], outputs: &mut [Vec<f32>],
         use_candidate: bool) -> Result<()> {
    for ((case, adapter), out) in cases.iter().zip(prepared).zip(outputs) {
        if use_candidate { adapter.run(black_box(case.input.as_ref()), case.m, black_box(out.as_mut_slice()))?; }
        else { baseline(case, out); }
        black_box(out.as_slice());
    }
    Ok(())
}
fn repeat_guard(outputs: &[Vec<f32>], expected: &[Vec<f32>]) -> Result<()> {
    ensure!(outputs.len() == expected.len(), "suite output count changed");
    for (actual, old) in outputs.iter().zip(expected) {
        ensure!(actual.len() == old.len() && actual.iter().zip(old).all(|(a,b)| a.to_bits()==b.to_bits()),
            "backend output changed across repetitions");
    }
    Ok(())
}
fn save_outputs(path: &Path, cases: &[Case<'_>], values: &[Vec<f32>]) -> Result<Value> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut offset = 0usize; let mut tensors = Vec::new();
    for (case, data) in cases.iter().zip(values) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        file.write_all(bytes)?;
        tensors.push(json!({"case":case.name,"shape":[case.m,case.n],"dtype":"F32",
            "byte_offset":offset,"byte_length":bytes.len(),"raw_sha256":sha(bytes)}));
        offset += bytes.len();
    }
    file.flush()?;
    Ok(json!({"path":path.file_name().unwrap().to_string_lossy(),"sha256":file_sha(path)?,
        "format":"concatenated little-endian FP32","tensors":tensors}))
}

fn run_phase(name: &str, cases: &[Case<'_>], args: &Args) -> Result<Value> {
    ensure!(rayon::current_num_threads() == THREADS, "wrong Rayon budget");
    // Prepared may be !Send: construct/use/drop entirely inside pool.install.
    let mut prepared = Vec::with_capacity(cases.len()); let mut preparations = Vec::new();
    for case in cases {
        let start = Instant::now();
        let adapter = candidate::Prepared::new(case.weight, case.k, case.n)?;
        let milliseconds = start.elapsed().as_secs_f64() * 1000.;
        preparations.push(json!({"case":case.name,"preparation_ms":milliseconds,
            "info":adapter.info(),"scope":"constructor only; includes any owned weight copy/packing; excludes mapped input load"}));
        prepared.push(adapter);
    }
    let mut controls: Vec<_> = cases.iter().map(|c| vec![f32::NAN;c.m*c.n]).collect();
    let mut candidates = controls.clone();
    suite(cases, &prepared, &mut controls, false)?;
    suite(cases, &prepared, &mut candidates, true)?;
    let mut accuracy = Vec::new();
    let mut original_control_ok = true; let mut arithmetic_bound_ok = true;
    for ((case, control), candidate) in cases.iter().zip(&controls).zip(&candidates) {
        let cpu = comparison(control, candidate)?;
        let saved = case.saved_cpu.map(|s| comparison(s, control)).transpose()?;
        if let Some(ref check) = saved { original_control_ok &= check["different_bits"] == 0; }
        let precise = oracle(case);
        let candidate_f64 = oracle_error(case, &precise, candidate);
        arithmetic_bound_ok &= candidate_f64["outside_gamma_k_plus_f64_uncertainty"] == 0;
        accuracy.push(json!({"case":case.name,"weight_key":case.weight_key,
            "shape":{"m":case.m,"k":case.k,"n":case.n},
            "input_raw_sha256":sha(bytemuck::cast_slice(case.input.as_ref())),
            "weight_raw_sha256":sha(bytemuck::cast_slice(case.weight)),
            "control_vs_saved_cpu":saved,"candidate_vs_control":cpu,
            "control_vs_gpu":case.gpu.map(|g| comparison(g,control)).transpose()?,
            "candidate_vs_gpu":case.gpu.map(|g| comparison(g,candidate)).transpose()?,
            "control_vs_f64":oracle_error(case,&precise,control),
            "candidate_vs_f64":candidate_f64,
            "gpu_vs_f64":case.gpu.map(|g| oracle_error(case,&precise,g)),
            "f64_values_raw_sha256":sha(bytemuck::cast_slice(&precise.values)),
            "f64_sum_abs_products_raw_sha256":sha(bytemuck::cast_slice(&precise.absolute_products))}));
    }
    let mut report = json!({"phase":name,"case_order":cases.iter().map(|c| &c.name).collect::<Vec<_>>(),
        "distinct_weight_tensor_bytes":cases.iter().map(|c|c.weight.len()*4).sum::<usize>(),
        "candidate_preparations":preparations,"accuracy":accuracy,
        "control_outputs":save_outputs(&args.output.join(format!("{name}-control-f32.bin")),cases,&controls)?,
        "candidate_outputs":save_outputs(&args.output.join(format!("{name}-candidate-f32.bin")),cases,&candidates)?,
        "saved_cpu_control_exact":original_control_ok,"candidate_dot_bound_passed":arithmetic_bound_ok,
        "model_or_corpus_quality_qualification":false});
    let accuracy_name = format!("{name}-accuracy.json");
    let accuracy_sha = write_json(&args.output.join(&accuracy_name), &report)?;
    report["accuracy_artifact"] = json!({"path":accuracy_name,"sha256":accuracy_sha});
    ensure!(original_control_ok, "original vocabulary projection does not reproduce saved CPU logits");
    ensure!(arithmetic_bound_ok, "candidate exceeds declared FP32-dot diagnostic envelope");

    let mut work = controls.clone();
    // Prime each backend and threadpool outside counted/timed intervals.
    suite(cases,&prepared,&mut work,false)?;
    let control_allocs = allocation_interval(|| suite(cases,&prepared,&mut work,false))?;
    repeat_guard(&work,&controls)?;
    suite(cases,&prepared,&mut work,true)?;
    let candidate_allocs = allocation_interval(|| suite(cases,&prepared,&mut work,true))?;
    repeat_guard(&work,&candidates)?;
    report["warm_allocations"] = json!({"control":control_allocs,"candidate":candidate_allocs});
    if args.measure {
        let mut runs = Vec::new();
        for (label, use_candidate) in [("control_before",false),("candidate",true),("control_after",false)] {
            let expected = if use_candidate { &candidates } else { &controls };
            for _ in 0..WARMUPS { suite(cases,&prepared,&mut work,use_candidate)?; repeat_guard(&work,expected)?; }
            let mut samples = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let start = Instant::now();
                for _ in 0..ITERATIONS { suite(cases,&prepared,&mut work,use_candidate)?; }
                samples.push(start.elapsed().as_secs_f64()*1000./ITERATIONS as f64);
                // Full output guard outside timing. This observes the final
                // suite iteration in each sample, not every overwritten result.
                repeat_guard(&work,expected)?;
            }
            let mut sorted=samples.clone();sorted.sort_by(f64::total_cmp);
            runs.push(json!({"label":label,"suite_ms_samples":samples,"median_suite_ms":sorted[SAMPLES/2],
                "all_final_sample_outputs_match_backend_accuracy_outputs":true}));
        }
        report["timing"] = json!({"threads":THREADS,"warmup_suites_per_arm":WARMUPS,
            "samples_per_arm":SAMPLES,"ordered_suite_iterations_per_sample":ITERATIONS,
            "arms":runs,"allocation_counters_enabled":false,
            "scope":"rotating complete ordered suite; excludes constructor, IO, F64 and output validation; includes per-call backend scratch/packing"});
    } else { report["timing"] = Value::Null; }
    Ok(report)
}

fn run(args: &Args) -> Result<()> {
    ensure!(cfg!(target_endian="little"), "little-endian tensor views required");
    kernels::Simd::Avx2.validate().map_err(anyhow::Error::msg)?;
    ensure!(sha(include_bytes!("baseline_kernels.rs")) == BASELINE_KERNELS_SHA, "baseline source changed");
    let binary = std::env::current_exe()?; let binary_sha = file_sha(&binary)?;
    let maps = [BoundMap::new(&args.root,FIXTURE)?,BoundMap::new(&args.root,SIDECAR)?,
        BoundMap::new(&args.root,CHECKPOINT)?,BoundMap::new(&args.root,CONFIG)?,BoundMap::new(&args.root,TRACE)?];
    let fixture=SafeTensors::deserialize(&maps[0].map)?;
    let metadata:Value=serde_json::from_slice(&maps[1].map)?;
    let weights=SafeTensors::deserialize(&maps[2].map)?;
    let config:Value=serde_json::from_slice(&maps[3].map)?;
    let trace=SafeTensors::deserialize(&maps[4].map)?;
    ensure!(metadata["output_sha256"]==FIXTURE.1 && metadata["dtype"]=="float32" && metadata["tf32"]==false,
        "fixture source contract changed");
    ensure!(config["dim"]==768 && config["vocab_size"]==65536 && config["ffn_dim"]==2304,
        "model dimensions changed");
    let eps=config["norm_eps"].as_f64().context("missing norm epsilon")? as f32;
    ensure!(eps.to_bits()==1.0e-5_f32.to_bits(), "final norm epsilon changed");
    let shapes=[("qkv",768,2048),("wo",1024,768),("w13",768,4608),("w2",2304,768)];
    let mut decode=Vec::new();
    for layer in [9,12,19,21] {
        for (op,k,n) in shapes {
            decode.push(fixture_case(&fixture,&metadata,&format!("decode.2.layer.{layer}.{op}"),layer,op,1,k,n)?);
        }
    }
    let pool=rayon::ThreadPoolBuilder::new().num_threads(THREADS).build()?;
    let hidden=tensor(&trace,"decode.1.layer.21.hidden",&[1,768])?;
    let norm=tensor(&weights,"norm.weight",&[768])?;
    let mut normalized=vec![0.;768];
    pool.install(||kernels::rms_norm(hidden,&mut normalized,768,eps,Some(norm)));
    let vocab_reconstruction=json!({"hidden_key":"decode.1.layer.21.hidden",
        "hidden_raw_sha256":sha(bytemuck::cast_slice(hidden)),"norm_raw_sha256":sha(bytemuck::cast_slice(norm)),
        "epsilon_bits":eps.to_bits(),"normalized_raw_sha256":sha(bytemuck::cast_slice(&normalized)),
        "operator":"unchanged frozen kernels::rms_norm; no model execution"});
    decode.push(Case{name:"cpu.decode.1.vocabulary".into(),weight_key:"output.weight".into(),m:1,k:768,n:65536,
        input:Cow::Owned(normalized),weight:tensor(&weights,"output.weight",&[65536,768])?,gpu:None,
        saved_cpu:Some(tensor(&trace,"decode.1.logits",&[65536])?)});
    ensure!(decode.len()==17 && decode.iter().map(|c|c.weight.len()*4).sum::<usize>()==324_009_984,
        "fixed rotated decode suite changed");
    let mut prefill=Vec::new();
    if args.prefill {
        for (op,k,n) in shapes {
            prefill.push(fixture_case(&fixture,&metadata,&format!("prefill.layer.0.{op}"),0,op,144,k,n)?);
        }
    }
    let start=json!({"kind":"actual-operands-matrix-backend-comparison-v1","status":"started",
        "binary":binary,"binary_sha256":binary_sha,"threads":THREADS,
        "measure":args.measure,"prefill":args.prefill,
        "input_sha256":maps.iter().map(|b|(b.path.to_string_lossy().into_owned(),Value::String(b.expected.into()))).collect::<serde_json::Map<_,_>>(),
        "source_sha256":{"main.rs":sha(include_bytes!("main.rs")),"candidate.rs":sha(include_bytes!("candidate.rs")),
            "baseline_kernels.rs":sha(include_bytes!("baseline_kernels.rs")),"Cargo.lock":sha(include_bytes!("Cargo.lock"))},
        "target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH,"debug_assertions":cfg!(debug_assertions),
        "baseline":"original Avx2 dispatch: custom dots for M<=8, independently dispatched gemm0.19 for M>8",
        "vocabulary_reconstruction":vocab_reconstruction,
        "limitations":["Saved synthetic-document activations are operator diagnostics, not quality calibration.",
            "Decode suite rotates309MiB distinct weights; it is not a model token or page latency measurement.",
            "Optional M144 prefill uses actual captured rows; no M6544/full-page performance claim.",
            "GPU exact bits are descriptive; independent F64 errors do not change the frozen full-model gates.",
            "Preparation time is separately recorded and must be charged/amortized explicitly in future integration.",
            "No cache flush, affinity guarantee, memory bandwidth or CPU stall cause is inferred."]});
    let start_sha=write_json(&args.output.join("start.json"),&start)?;
    let mut phases=Vec::new();
    phases.push(pool.install(||run_phase("decode",&decode,args))?);
    if args.prefill { phases.push(pool.install(||run_phase("prefill144",&prefill,args))?); }
    for binding in &maps { binding.check()?; }
    ensure!(file_sha(&binary)?==binary_sha,"binary changed during run");
    let mut artifacts=serde_json::Map::new();
    for phase in &phases {
        for key in ["control_outputs","candidate_outputs"] {
            let file=phase[key]["path"].as_str().context("missing output artifact")?;
            let expected=phase[key]["sha256"].as_str().context("missing output digest")?;
            ensure!(file_sha(&args.output.join(file))?==expected,"saved output changed");
            artifacts.insert(file.into(),json!(expected));
        }
        let file=phase["accuracy_artifact"]["path"].as_str().context("missing accuracy artifact")?;
        let expected=phase["accuracy_artifact"]["sha256"].as_str().context("missing accuracy digest")?;
        ensure!(file_sha(&args.output.join(file))?==expected,"accuracy artifact changed");
        artifacts.insert(file.into(),json!(expected));
    }
    ensure!(file_sha(&args.output.join("start.json"))?==start_sha,"startup artifact changed");
    artifacts.insert("start.json".into(),json!(start_sha));
    let mut result=start;
    result["status"]=json!("complete_operator_diagnostic");result["phases"]=json!(phases);
    result["output_artifact_sha256"]=json!(artifacts);
    result["input_binary_closure"]=json!(true);
    result["performance_promotion"]=json!(false);
    write_json(&args.output.join("report.json"),&result)?;
    println!("{}",json!({"status":result["status"],"report_sha256":file_sha(&args.output.join("report.json"))?}));
    Ok(())
}

fn main() -> Result<()> {
    let args=Args::parse();
    ensure!(!args.output.exists(),"preserve existing output directory");
    std::fs::create_dir_all(&args.output)?;
    if let Err(error)=run(&args) {
        write_json(&args.output.join("failed.json"),&json!({"status":"failed","error":format!("{error:#}"),
            "performance_promotion":false,"partial_artifacts_preserved":true}))?;
        return Err(error);
    }
    Ok(())
}
