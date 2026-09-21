//! Included only inside model::fp32_crossover in an isolated test-only copy.
use super::*;
use crate::trace::TensorTrace;
use std::collections::BTreeMap;
use std::path::PathBuf;

include!("fp32_crossover_forward.rs");

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn saved_name(tensors: &SafeTensors<'_>, name: &str) -> Result<String> {
    let candidates = [name.to_owned(), format!("prefill.{name}")];
    let present = candidates.into_iter().filter(|key| tensors.tensor(key).is_ok()).collect::<Vec<_>>();
    ensure!(present.len() == 1, "missing or ambiguous saved prefill stage {name}");
    Ok(present[0].clone())
}

fn f32_tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::F32 && tensor.shape() == shape, "wrong {name} dtype/shape");
    Ok(tensor.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn ids(tensors: &SafeTensors<'_>, name: &str, len: usize) -> Result<Vec<u32>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.shape() == [len], "wrong {name} shape");
    match tensor.dtype() {
        Dtype::I64 => tensor.data().chunks_exact(8)
            .map(|b| Ok(u32::try_from(i64::from_le_bytes(b.try_into().unwrap()))?)).collect(),
        Dtype::U32 => Ok(tensor.data().chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()),
        _ => bail!("unsupported integer dtype for {name}"),
    }
}

fn exact(trace: &TensorTrace, cpu: &SafeTensors<'_>, name: &str) -> Result<serde_json::Value> {
    let (shape, actual) = trace.tensors.get(&format!("cpu_state.{name}")).context("missing control stage")?;
    let expected = cpu.tensor(&saved_name(cpu, name)?)?;
    ensure!(expected.dtype() == Dtype::F32 && expected.shape() == shape.as_slice(), "control shape/dtype");
    let mismatch = actual.iter().zip(expected.data().chunks_exact(4))
        .filter(|(a, b)| a.to_bits() != u32::from_le_bytes((*b).try_into().unwrap())).count();
    Ok(serde_json::json!({"stage":name,"shape":shape,"elements":actual.len(),
        "bit_mismatches":mismatch,"passed":mismatch==0,"expected_raw_sha256":digest(expected.data()),
        "actual_raw_sha256":digest(bytemuck::cast_slice(actual))}))
}

#[test]
#[ignore = "requires reviewed frozen plan, isolated binary and quiet-window release"]
fn capture_cpu_state_crossover() -> Result<()> {
    ensure!(cfg!(target_os="windows") && cfg!(target_endian="little"), "original baseline is native Windows little-endian");
    let root = PathBuf::from(std::env::var("FOCR_CROSSOVER_ROOT")?);
    let output = PathBuf::from(std::env::var("FOCR_CROSSOVER_OUTPUT")?);
    let plan_path = PathBuf::from(std::env::var("FOCR_CROSSOVER_PLAN")?);
    let plan_bytes = std::fs::read(&plan_path)?;
    let plan_sha = digest(&plan_bytes);
    ensure!(plan_sha == std::env::var("FOCR_CROSSOVER_PLAN_SHA256")?, "plan hash");
    let plan: serde_json::Value = serde_json::from_slice(&plan_bytes)?;
    ensure!(plan["kind"] == "fp32-frozen-state-crossover-v1", "plan kind");
    ensure!(plan["runtime"]["rows"] == 144 && plan["runtime"]["layer"] == 8
        && plan["runtime"]["next_layer"] == 9 && plan["runtime"]["rust_threads"] == 4
        && plan["runtime"]["rust_backend"] == "avx2", "fixed runtime");
    ensure!(!output.exists(), "refusing existing Rust arm output");
    std::fs::create_dir(&output)?;
    let load = |role: &str, pin: &str| -> Result<Vec<u8>> {
        let entry = &plan["inputs"][role];
        ensure!(entry["sha256"] == pin, "fixed input pin");
        let path = Path::new(entry["path"].as_str().context("input path")?);
        ensure!(!path.is_absolute() && !path.components().any(|x| matches!(x, std::path::Component::ParentDir)), "relative input path");
        let bytes = std::fs::read(root.join(path))?;
        ensure!(digest(&bytes) == pin, "input changed: {role}");
        Ok(bytes)
    };
    let cpu_bytes = load("cpu_trace", "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309")?;
    let gpu_bytes = load("gpu_trace", "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4")?;
    let cpu = SafeTensors::deserialize(&cpu_bytes)?;
    let gpu = SafeTensors::deserialize(&gpu_bytes)?;
    let model = Model::load(root.join("artifacts/model"))?;
    let c = model.config();
    ensure!(c.dim == 768 && c.n_heads == 16 && c.n_kv_heads == 8 && c.head_dim == 64
        && c.ffn_dim == 2304 && c.n_layers == 22, "fixed Falcon architecture");
    let tokens = ids(&gpu, "tokens", 144)?;
    let teachers = ids(&gpu, "teacher_tokens", 17)?;
    let spatial = f32_tensor(&gpu, "pos_hw", &[144,2])?;
    let patch_hw = tokens.iter().zip(spatial.chunks_exact(2))
        .filter_map(|(&token, hw)| (token == c.img_id).then_some([hw[0], hw[1]])).collect::<Vec<_>>();
    let (t, hw) = positions(&tokens, &patch_hw, c)?;
    ensure!(t.iter().map(|&x| x as u32).eq(ids(&gpu,"pos_t",144)?), "temporal positions differ");
    for (&a, &b) in hw.iter().flatten().zip(&spatial) {
        ensure!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "spatial position mismatch");
    }
    let (image_start, image_end) = image_range(&tokens, c)?;
    let mut hidden = f32_tensor(&cpu, &saved_name(&cpu, "layer.7.hidden")?, &[144,768])?;
    ensure!(hidden.iter().all(|x| x.is_finite()), "nonfinite saved state");
    let original_input_sha = digest(bytemuck::cast_slice(&hidden));
    let simd = kernels::Simd::Avx2;
    simd.validate().map_err(anyhow::Error::msg)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4)
        .start_handler(|_| crate::numerical_diagnostics::set_thread_rms_variant(0)).build()?;
    let mut trace = TensorTrace::default();
    pool.install(|| -> Result<()> {
        let mut session = Session::new(c, tokens.len()+teachers.len(), tokens.len(), image_start, image_end,
                                       simd, CacheLayout::Expanded)?;
        model.crossover_segment(&mut hidden, &t, &hw, &mut session, &mut trace, "cpu_state")
    })?;
    let expected_stages = plan["stages"].as_object().context("stage shapes")?;
    ensure!(trace.tensors.len() == expected_stages.len() && expected_stages.len() == 17, "capture inventory");
    let mut tensor_records = BTreeMap::new();
    for (name, dims) in expected_stages {
        let full_name = format!("cpu_state.{name}");
        let (shape, values) = trace.tensors.get(&full_name).context("missing capture")?;
        let wanted = dims.as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>();
        ensure!(*shape == wanted && values.iter().all(|x| x.is_finite()), "invalid stage {name}");
        tensor_records.insert(full_name, serde_json::json!({"dtype":"F32","shape":shape,
            "elements":values.len(),"raw_sha256":digest(bytemuck::cast_slice(values))}));
    }
    let controls = ["layer.8.q","layer.8.k","layer.8.v","layer.8.attention","layer.8.hidden","layer.9.v"]
        .into_iter().map(|name| exact(&trace,&cpu,name)).collect::<Result<Vec<_>>>()?;
    let passed = controls.iter().all(|x| x["passed"] == true);
    let tensors_path = output.join("tensors.safetensors");
    trace.save(&tensors_path)?;
    let report = serde_json::json!({"kind":"fp32-crossover-rust-arm-v1",
        "status":if passed {"control_exact"} else {"rejected_original_control_mismatch"},
        "plan_sha256":plan_sha,"branch":"cpu_state","controls":controls,"tensors":tensor_records,
        "input_raw_sha256":original_input_sha,"tensor_file_sha256":digest(&std::fs::read(&tensors_path)?),
        "test_binary_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "runtime":{"threads":4,"backend":"avx2","precision":"fp32","cache_capacity":161},
        "source_sha256":{"arm":digest(include_bytes!("fp32_crossover_arm.rs")),
            "adapter":digest(include_bytes!("fp32_crossover_forward.rs")),"kernels":digest(include_bytes!("kernels.rs"))},
        "arithmetic_changed":false,"control_is_saved_payload_not_new_full_graph":true,
        "qualification":"Isolated observed segment only. No hidden-stage policy closure, generation, performance or startup provenance claim."});
    std::fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    ensure!(passed,"original Rust control mismatch; saved outputs rejected");
    Ok(())
}
