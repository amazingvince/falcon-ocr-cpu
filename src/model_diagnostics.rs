//! Diagnostic interventions, compiled only for tests. They never select a
//! production backend or modify the frozen numerical acceptance policy.
use super::*;
use crate::trace::TensorTrace;
use rayon::prelude::*;

fn cuda_tree_rms(
    input: &[f32],
    width: usize,
    fused_squares: bool,
    rounded_rsqrt: bool,
) -> Vec<f32> {
    assert!(width == 64 || width == 768);
    let mut output = vec![0.; input.len()];
    for (source, target) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let mut lanes = [0.0f32; 128];
        for (thread, sum) in lanes.iter_mut().enumerate() {
            for start in (thread * 4..width).step_by(512) {
                for &x in &source[start..start + 4] {
                    *sum = if fused_squares {
                        x.mul_add(x, *sum)
                    } else {
                        *sum + x * x
                    };
                }
            }
        }
        for warp in lanes.chunks_exact_mut(32) {
            for offset in [16, 8, 4, 2, 1] {
                for lane in 0..offset {
                    warp[lane] += warp[lane + offset];
                }
            }
        }
        let sum = (lanes[0] + lanes[64]) + (lanes[32] + lanes[96]);
        let variance = sum / width as f32 + f32::EPSILON;
        let scale = if rounded_rsqrt {
            (1.0 / (variance as f64).sqrt()) as f32
        } else {
            variance.sqrt().recip()
        };
        for (y, &x) in target.iter_mut().zip(source) {
            *y = x * scale;
        }
    }
    output
}

fn difference(actual: &[f32], expected: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let mut maximum = 0.0f64;
    let mut squared = 0.0f64;
    let mut peak = 0.0f64;
    let mut index = 0;
    let mut differing = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        let error = (a as f64 - b as f64).abs();
        if error > maximum {
            maximum = error;
            index = i;
        }
        squared += error * error;
        peak = peak.max((b as f64).abs());
        differing += usize::from(a.to_bits() != b.to_bits());
    }
    serde_json::json!({"elements":actual.len(), "differing_elements":differing,
        "max_abs":maximum, "rms_error":(squared/actual.len() as f64).sqrt(),
        "reference_peak":peak, "max_abs_over_peak":maximum/peak.max(f64::MIN_POSITIVE),
        "worst_index":index, "worst_actual":actual[index], "worst_expected":expected[index]})
}

#[cfg(target_arch = "x86_64")]
fn uninterrupted_fma_linear(
    input: &[f32],
    weight: &[f32],
    width: usize,
    channels: usize,
) -> Vec<f32> {
    assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"));
    assert_eq!(channels % 32, 0);
    assert_eq!(weight.len(), width * channels);
    assert_eq!(input.len() % width, 0);
    // Diagnostic packing exposes output channels as SIMD lanes, with one
    // uninterrupted ascending-K FMA chain per result instead of split-K sums.
    let mut packed = vec![0.; weight.len()];
    for block in 0..channels / 32 {
        for k in 0..width {
            for lane in 0..32 {
                packed[(block * width + k) * 32 + lane] = weight[(block * 32 + lane) * width + k];
            }
        }
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn row(input: &[f32], packed: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;
        for (block, chunk) in output.chunks_exact_mut(32).enumerate() {
            let mut sums = [_mm256_setzero_ps(); 4];
            for (k, &value) in input.iter().enumerate() {
                let x = _mm256_set1_ps(value);
                for (group, sum) in sums.iter_mut().enumerate() {
                    // SAFETY: caller checked exact packed dimensions and 32-wide blocks.
                    let w = unsafe {
                        _mm256_loadu_ps(
                            packed
                                .as_ptr()
                                .add((block * input.len() + k) * 32 + group * 8),
                        )
                    };
                    *sum = _mm256_fmadd_ps(x, w, *sum);
                }
            }
            for (group, sum) in sums.into_iter().enumerate() {
                // SAFETY: each exclusive output block contains 32 values.
                unsafe {
                    _mm256_storeu_ps(chunk.as_mut_ptr().add(group * 8), sum);
                }
            }
        }
    }
    let mut output = vec![0.; input.len() / width * channels];
    output
        .par_chunks_mut(channels)
        .zip(input.par_chunks(width))
        .for_each(|(out, input)| {
            // SAFETY: AVX2/FMA and all dimensions were validated above.
            unsafe {
                row(input, &packed, out);
            }
        });
    output
}

#[test]
#[ignore = "diagnostic requires independently exported pinned GPU layer fixtures"]
fn compare_equal_input_layer_substages() -> Result<()> {
    let output = std::path::PathBuf::from(
        std::env::var("FOCR_DIAGNOSTIC_OUTPUT_DIR").context("set FOCR_DIAGNOSTIC_OUTPUT_DIR")?,
    );
    std::fs::create_dir_all(&output)?;
    let bytes = std::fs::read("artifacts/reference/layer-operators-fp32.safetensors")?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(
        "artifacts/reference/layer-operators-fp32.json",
    )?)?;
    ensure!(
        metadata["output_sha256"].as_str() == Some(&format!("{:x}", Sha256::digest(&bytes))),
        "fixture hash"
    );
    let values = |name: &str| -> Result<Vec<f32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(tensor.dtype() == Dtype::F32, "expected F32 {name}");
        Ok(tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let token_tensor = tensors.tensor("tokens")?;
    let tokens = token_tensor
        .data()
        .chunks_exact(8)
        .map(|b| u32::try_from(i64::from_le_bytes(b.try_into().unwrap())))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let model = Model::load("artifacts/model")?;
    let c = model.config();
    let (image_start, image_end) = image_range(&tokens, c)?;
    let mut report = Vec::new();
    // Reset every substage to actual GPU input, then separately measure the
    // complete FFN chain. These interventions never alter production inference.
    for threads in [4, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        pool.install(|| -> Result<()> {
            for case in metadata["cases"].as_array().context("fixture cases")? {
                let name = case["name"].as_str().unwrap();
                let layer = &model.layers[case["layer"].as_u64().unwrap() as usize];
                let rows = case["rows"].as_u64().unwrap() as usize;
                let offset = case["query_offset"].as_u64().unwrap() as usize;
                let simd = kernels::Simd::Avx2;
                let get = |suffix: &str| values(&format!("{name}.{suffix}"));
                let mut stages = serde_json::Map::new();
                let mut add = |stage: &str, actual: &[f32], expected_key: &str| -> Result<()> {
                    stages.insert(stage.to_owned(), difference(actual, &get(expected_key)?));
                    Ok(())
                };
                let input = get("input")?;
                let residual = get("attention_residual.expected")?;
                for (stage, source, expected) in [
                    ("attention_norm", &input, "attention_norm.expected"),
                    ("ffn_norm", &residual, "ffn_norm.expected"),
                ] {
                    let mut actual = vec![0.; source.len()];
                    kernels::rms_norm(source, &mut actual, c.dim, f32::EPSILON, None);
                    add(stage, &actual, expected)?;
                    for fused_squares in [false,true] { for rounded_rsqrt in [false,true] {
                        let candidate = cuda_tree_rms(source,c.dim,fused_squares,rounded_rsqrt);
                        add(&format!("{stage}_cuda_tree_fma{fused_squares}_rounded_rsqrt{rounded_rsqrt}"), &candidate, expected)?;
                    } }
                }
                for (stage, source, weight, input_width, output_width, expected) in [
                    (
                        "qkv",
                        "attention_norm.expected",
                        &layer.qkv,
                        c.dim,
                        c.query_dim() + 2 * c.kv_dim(),
                        "qkv.expected",
                    ),
                    (
                        "attention_projection",
                        "attention.scaled",
                        &layer.wo,
                        c.query_dim(),
                        c.dim,
                        "attention_projection.expected",
                    ),
                    (
                        "w13",
                        "ffn_norm.expected",
                        &layer.w13,
                        c.dim,
                        2 * c.ffn_dim,
                        "w13.expected",
                    ),
                    (
                        "w2",
                        "gate.expected",
                        &layer.w2,
                        c.ffn_dim,
                        c.dim,
                        "w2.expected",
                    ),
                ] {
                    let mut actual = vec![0.; rows * output_width];
                    kernels::linear_with_simd(
                        &get(source)?,
                        rows,
                        input_width,
                        model.w(weight),
                        output_width,
                        &mut actual,
                        simd,
                    );
                    add(stage, &actual, expected)?;
                }
                let mut gate = vec![0.; rows * c.ffn_dim];
                kernels::squared_relu_gate(&get("w13.expected")?, &mut gate);
                add("gate", &gate, "gate.expected")?;
                let attention_sum: Vec<_> = input
                    .iter()
                    .zip(get("attention_projection.expected")?)
                    .map(|(&a, b)| a + b)
                    .collect();
                add(
                    "attention_residual",
                    &attention_sum,
                    "attention_residual.expected",
                )?;
                let final_sum: Vec<_> = residual
                    .iter()
                    .zip(get("w2.expected")?)
                    .map(|(&a, b)| a + b)
                    .collect();
                add("final_residual", &final_sum, "hidden.expected")?;
                let q = get("attention.q")?;
                let k = get("attention.k")?;
                let v = get("attention.v")?;
                let kv_len = offset + rows;
                ensure!(
                    k.len() >= kv_len * c.query_dim() && v.len() >= kv_len * c.query_dim(),
                    "cache length"
                );
                let mut attention = vec![0.; q.len()];
                kernels::attention_with_simd(
                    &q,
                    &k[..kv_len * c.query_dim()],
                    &v[..kv_len * c.query_dim()],
                    rows,
                    kv_len,
                    c.n_heads,
                    c.head_dim,
                    offset,
                    image_start,
                    image_end,
                    model.w(&layer.sinks),
                    &mut attention,
                    simd,
                );
                add("attention", &attention, "attention.scaled")?;
                let mut norm = vec![0.; residual.len()];
                let mut w13 = vec![0.; rows * 2 * c.ffn_dim];
                let mut w2 = vec![0.; residual.len()];
                kernels::rms_norm(&residual, &mut norm, c.dim, f32::EPSILON, None);
                kernels::linear_with_simd(
                    &norm,
                    rows,
                    c.dim,
                    model.w(&layer.w13),
                    2 * c.ffn_dim,
                    &mut w13,
                    simd,
                );
                kernels::squared_relu_gate(&w13, &mut gate);
                kernels::linear_with_simd(
                    &gate,
                    rows,
                    c.ffn_dim,
                    model.w(&layer.w2),
                    c.dim,
                    &mut w2,
                    simd,
                );
                for (value, input) in w2.iter_mut().zip(&residual) {
                    *value += input;
                }
                add("ffn_chain", &w2, "hidden.expected")?;
                #[cfg(target_arch = "x86_64")]
                {
                    let actual = uninterrupted_fma_linear(
                        &get("gate.expected")?,
                        model.w(&layer.w2),
                        c.ffn_dim,
                        c.dim,
                    );
                    add("w2_uninterrupted_fma", &actual, "w2.expected")?;
                }
                report.push(serde_json::json!({"name":name,"threads":threads,"stages":stages}));
            }
            Ok(())
        })?;
    }
    let report = serde_json::json!({"purpose":"Equal-input FP32 substage diagnostics; no changed acceptance bounds",
        "fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),"weights_sha256":model.weights_sha256(),
        "diagnostic_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("model_diagnostics.rs"))),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("kernels.rs"))),
        "backend":"avx2", "platform":std::env::consts::OS, "cases":report});
    std::fs::write(
        output.join("equal-input-substages.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("Saved equal-input diagnostics to {}", output.display());
    Ok(())
}

#[test]
#[ignore = "diagnostic requires pinned GPU trace and explicit output directory"]
fn trace_projector_numerical_interventions() -> Result<()> {
    trace_numerical_interventions(&["production", "fp64_projector", "gpu_embedding"])
}

#[test]
#[ignore = "diagnostic requires explicit output directory and pinned GPU reference"]
fn trace_cuda_rms_interventions() -> Result<()> {
    trace_numerical_interventions(&["production", "cuda_rms", "cuda_rms_rounded_rsqrt"])
}

#[test]
#[ignore = "diagnostic requires pinned GPU trace and explicit verified AOCL library"]
fn trace_aocl_prefill_w2_intervention() -> Result<()> {
    trace_numerical_interventions(&["production", "aocl_prefill_w2", "production_after"])
}

fn trace_numerical_interventions(interventions: &[&str]) -> Result<()> {
    let output = std::path::PathBuf::from(
        std::env::var("FOCR_DIAGNOSTIC_OUTPUT_DIR")
            .context("set FOCR_DIAGNOSTIC_OUTPUT_DIR for diagnostic trace artifacts")?,
    );
    std::fs::create_dir_all(&output)?;
    let fixture_path = Path::new("artifacts/reference/smoke-fp32/trace.safetensors");
    let bytes = std::fs::read(fixture_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let f32s = |name: &str| -> Result<Vec<f32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(tensor.dtype() == Dtype::F32, "expected F32 {name}");
        Ok(tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let ids = |name: &str| -> Result<Vec<u32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(tensor.dtype() == Dtype::I64, "expected I64 {name}");
        tensor
            .data()
            .chunks_exact(8)
            .map(|b| Ok(u32::try_from(i64::from_le_bytes(b.try_into().unwrap()))?))
            .collect()
    };
    let model = Model::load("artifacts/model")?;
    let c = model.config();
    let tokens = ids("tokens")?;
    let teachers = ids("teacher_tokens")?;
    ensure!(!teachers.is_empty(), "teacher tokens are required");
    let t = ids("pos_t")?
        .into_iter()
        .map(|v| v as usize)
        .collect::<Vec<_>>();
    let hw = f32s("pos_hw")?
        .chunks_exact(2)
        .map(|v| [v[0], v[1]])
        .collect::<Vec<_>>();
    let patches = f32s("patches")?;
    let gpu_embedding = f32s("embedding")?;
    ensure!(
        gpu_embedding.len() == tokens.len() * c.dim,
        "GPU embedding shape"
    );
    let (image_start, image_end) = image_range(&tokens, c)?;
    let simd = kernels::Simd::Avx2;
    simd.validate().map_err(anyhow::Error::msg)?;
    let mut report = Vec::new();
    let aocl = if interventions.contains(&"aocl_prefill_w2") {
        Some((
            std::path::PathBuf::from(std::env::var("FOCR_AOCL_LIBRARY")?).canonicalize()?,
            std::env::var("FOCR_AOCL_SHA256")?,
        ))
    } else {
        None
    };
    for &intervention in interventions {
        let rms_variant = match intervention {
            "cuda_rms" => 1,
            "cuda_rms_rounded_rsqrt" => 2,
            _ => 0,
        };
        let worker_aocl = if intervention == "aocl_prefill_w2" {
            aocl.clone()
        } else {
            None
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .start_handler(move |_| {
                crate::numerical_diagnostics::set_thread_rms_variant(rms_variant);
                if let Some((path, hash)) = &worker_aocl {
                    crate::numerical_diagnostics::set_thread_aocl_prefill_w2(path, hash)
                        .expect("load explicitly hash-verified AOCL diagnostic library");
                }
            })
            .build()?;
        let record = pool.install(|| -> Result<_> {
            let mut hidden = Vec::new();
            model.embed(&tokens, Some(&patches), simd, &mut hidden)?;
            if intervention == "gpu_embedding" {
                hidden.copy_from_slice(&gpu_embedding);
            } else if intervention == "fp64_projector" {
                let weights = model.w(&model.projector);
                let mut features = vec![0f32; patches.len() / c.patch_dim() * c.dim];
                features
                    .par_chunks_mut(c.dim)
                    .zip(patches.par_chunks(c.patch_dim()))
                    .for_each(|(dst, input)| {
                        for (value, weight) in
                            dst.iter_mut().zip(weights.chunks_exact(c.patch_dim()))
                        {
                            // Independent FP64 mathematical accumulation, rounded
                            // once to FP32. This is a diagnostic, not FP32 inference.
                            *value = input
                                .iter()
                                .zip(weight)
                                .map(|(&x, &w)| x as f64 * w as f64)
                                .sum::<f64>() as f32;
                        }
                    });
                let mut patch = 0;
                for (&token, row) in tokens.iter().zip(hidden.chunks_exact_mut(c.dim)) {
                    if token == c.img_id {
                        row.copy_from_slice(&features[patch * c.dim..(patch + 1) * c.dim]);
                        patch += 1;
                    }
                }
            }
            let mut session = Session::new(
                c,
                tokens.len() + teachers.len(),
                tokens.len(),
                image_start,
                image_end,
                simd,
                CacheLayout::Expanded,
            )?;
            let mut trace = TensorTrace::default();
            let argmax = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .unwrap()
                    .0 as u32
            };
            let logits =
                model.forward(&mut hidden, &t, &hw, &mut session, &mut trace, "prefill")?;
            let mut chosen = vec![argmax(logits)];
            for (step, &teacher) in teachers.iter().take(teachers.len() - 1).enumerate() {
                model.embed(&[teacher], None, simd, &mut hidden)?;
                let position = session.next_position;
                let logits = model.forward(
                    &mut hidden,
                    &[position],
                    &[[f32::NAN; 2]],
                    &mut session,
                    &mut trace,
                    &format!("decode.{step}"),
                )?;
                chosen.push(argmax(logits));
            }
            let path = output.join(format!("{intervention}.safetensors"));
            trace.save(&path)?;
            Ok(
                serde_json::json!({"intervention":intervention,"trace_path":path,
                "tensors":trace.tensors.len(),"same_prefix_argmax":chosen,
                "argmax_matches_teacher":chosen == teachers,"teacher_forced":true}),
            )
        })?;
        report.push(record);
    }
    let report = serde_json::json!({"purpose":"Isolate numerical sensitivity to independently defined operations; diagnostic interventions do not qualify a runtime",
        "fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),"weights_sha256":model.weights_sha256(),
        "model_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("model.rs"))),
        "diagnostic_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("model_diagnostics.rs"))),
        "numerical_dispatch_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("numerical_diagnostics.rs"))),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("kernels.rs"))),
        "aocl_binding_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../examples/support/aocl_dynamic.rs"))),
        "aocl_library":aocl.as_ref().map(|(path, hash)|serde_json::json!({"path":path,"sha256":hash,"threading":"OpenMP off; single-thread library"})),
        "test_binary_sha256":format!("{:x}",Sha256::digest(std::fs::read(std::env::current_exe()?)?)),
        "backend":"avx2","threads":4,"platform":std::env::consts::OS,"interventions":report});
    std::fs::write(
        output.join("interventions.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
