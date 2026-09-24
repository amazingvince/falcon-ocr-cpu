//! Independent GPU fixture comparison for explicit BF16 graph building blocks.
use anyhow::{Result, ensure};
use falcon_ocr::bf16_ops::{self, NormWorkspace};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn read(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<bf16>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::BF16, "{name}: expected BF16");
    Ok(tensor
        .data()
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
        .collect())
}
fn compare(expected: &[bf16], actual: &[bf16]) -> Value {
    assert_eq!(expected.len(), actual.len());
    let mut max_abs = 0.0_f32;
    let mut different = 0;
    let mut nonfinite = 0;
    let mut max_ulp = 0;
    let mut worst_index = None;
    let ordered = |value: bf16| {
        let bits = value.to_bits();
        if bits & 0x8000 != 0 {
            !bits
        } else {
            bits ^ 0x8000
        }
    };
    for (index, (a, b)) in expected.iter().zip(actual).enumerate() {
        different += usize::from(a.to_bits() != b.to_bits());
        nonfinite += usize::from(!a.is_finite() || !b.is_finite());
        max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
        if a.is_finite() && b.is_finite() {
            let ulp = ordered(*a).abs_diff(ordered(*b));
            if ulp > max_ulp {
                max_ulp = ulp;
                worst_index = Some(index);
            }
        }
    }
    json!({"elements":actual.len(),"different_bits":different,"maximum_absolute_error":max_abs,
        "maximum_bf16_ulp_distance":max_ulp,"worst_ulp_index":worst_index,"nonfinite_pairs":nonfinite})
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() -> Result<()> {
    let fixture = "artifacts/reference/bf16-operators.safetensors";
    let bytes = std::fs::read(fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let affine_fixture = "artifacts/reference/attention-operators-bf16.safetensors";
    let affine_bytes = std::fs::read(affine_fixture)?;
    let affine_tensors = SafeTensors::deserialize(&affine_bytes)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let operators = pool.install(|| -> Result<Vec<Value>> {
        let mut reports = Vec::new();
        let mut scratch = NormWorkspace::default();
        for (name, width) in [
            ("rms.width768.actual", 768),
            ("rms.width64.actual", 64),
            ("rms.width768.small", 768),
            ("rms.width64.small", 64),
        ] {
            let input = read(&tensors, &format!("{name}.input"))?;
            let expected = read(&tensors, &format!("{name}.implicit"))?;
            let mut actual = vec![bf16::ZERO; input.len()];
            bf16_ops::rms_norm(&input, &mut actual, width, f32::EPSILON, None, &mut scratch);
            reports.push(json!({"operator":name,"comparison":compare(&expected,&actual)}));
        }
        for name in ["gate.layer.0", "gate.layer.12"] {
            let input = read(&tensors, &format!("{name}.input"))?;
            let expected = read(&tensors, &format!("{name}.expected"))?;
            let mut actual = vec![bf16::ZERO; expected.len()];
            bf16_ops::squared_relu_gate(&input, &mut actual);
            reports.push(json!({"operator":name,"comparison":compare(&expected,&actual)}));
        }
        let weight = read(&affine_tensors, "final_norm.weight")?;
        for name in [
            "prefill.final_norm",
            "decode.2.final_norm",
            "decode.6.final_norm",
        ] {
            let input = read(&affine_tensors, &format!("{name}.input"))?;
            let expected = read(&affine_tensors, &format!("{name}.expected"))?;
            let mut actual = vec![bf16::ZERO; expected.len()];
            bf16_ops::rms_norm(
                &input,
                &mut actual,
                weight.len(),
                1e-5,
                Some(&weight),
                &mut scratch,
            );
            reports.push(json!({"operator":name,"comparison":compare(&expected,&actual)}));
        }
        Ok(reports)
    })?;
    let report = json!({"fixture":fixture,"fixture_sha256":hash(&bytes),
        "affine_fixture":affine_fixture,"affine_fixture_sha256":hash(&affine_bytes),
        "source_sha256":hash(include_bytes!("../src/bf16_ops.rs")),"harness_sha256":hash(include_bytes!("bf16_ops_probe.rs")),
        "platform":std::env::consts::OS,"threads":2,"operators":operators,
        "qualification":"Isolated RMSNorm and gate semantics only; no full-model numerical or performance qualification."});
    let path = format!("reference/bf16-ops-{}.json", std::env::consts::OS);
    std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
