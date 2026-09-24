//! Rotary factors, token positions and the row-wise helpers of the forward pass.
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

use crate::{config::ModelConfig, kernels};

use super::PARALLEL_ROWS;

pub(crate) fn temporal_factors(c: &ModelConfig) -> Vec<[f32; 2]> {
    let pairs = c.head_dim / 4;
    let mut result = Vec::with_capacity(c.max_seq_len * pairs);
    for t in 0..c.max_seq_len {
        for pair in 0..pairs {
            let freq = 1. / c.rope_theta.powf((2 * pair) as f32 / (c.head_dim / 2) as f32);
            let (sin, cos) = (t as f32 * freq).sin_cos();
            result.push([cos, sin]);
        }
    }
    result
}

pub(crate) fn rotary_factors(
    c: &ModelConfig,
    t: &[usize],
    hw: &[[f32; 2]],
    golden: &[f32],
    temporal: &[[f32; 2]],
    result: &mut Vec<[f32; 2]>,
) {
    let pairs = c.head_dim / 2;
    let temporal_pairs = pairs / 2;
    result.resize(t.len() * c.n_heads * pairs, [1., 0.]);
    let fill = |row: usize, factors: &mut [[f32; 2]]| {
        for head in 0..c.n_heads {
            for pair in 0..pairs {
                if pair < temporal_pairs {
                    factors[head * pairs + pair] = temporal[t[row] * temporal_pairs + pair];
                    continue;
                }
                let angle = if hw[row][0].is_finite() && hw[row][1].is_finite() {
                    let index = (head * temporal_pairs + pair - temporal_pairs) * 2;
                    hw[row][0] * golden[index] + hw[row][1] * golden[index + 1]
                } else {
                    0.
                };
                let (sin, cos) = angle.sin_cos();
                factors[head * pairs + pair] = [cos, sin];
            }
        }
    };
    let width = c.n_heads * pairs;
    if t.len() >= PARALLEL_ROWS {
        result
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, factors)| fill(row, factors));
    } else {
        for (row, factors) in result.chunks_mut(width).enumerate() {
            fill(row, factors);
        }
    }
}

/// `h += projected`, row-parallel for prefill-sized inputs.
/// `rms_norm`'s factor of every `width`-wide row of `h` (eps `f32::EPSILON`, no weight).
pub(super) fn row_scales(h: &[f32], width: usize, out: &mut Vec<f32>) {
    out.resize(h.len() / width, 0.0);
    out.par_iter_mut()
        .zip(h.par_chunks(width))
        .for_each(|(scale, row)| *scale = kernels::rms_scale(row, width, f32::EPSILON));
}

pub(super) fn add_residual(h: &mut [f32], projected: &[f32], dim: usize) {
    if h.len() >= PARALLEL_ROWS * dim {
        h.par_chunks_mut(dim).zip(projected.par_chunks(dim)).for_each(|(h, a)| {
            for (x, a) in h.iter_mut().zip(a) {
                *x += a;
            }
        });
    } else {
        for (x, a) in h.iter_mut().zip(projected) {
            *x += a;
        }
    }
}

pub(crate) fn image_range(tokens: &[u32], c: &ModelConfig) -> Result<(usize, usize)> {
    let start = tokens
        .iter()
        .position(|&x| x == c.image_cls_token_id)
        .context("missing image class token")?;
    let end = tokens
        .iter()
        .position(|&x| x == c.img_end_id)
        .context("missing image end token")?;
    ensure!(start < end, "invalid image token range");
    ensure!(
        tokens.iter().filter(|&&x| x == c.image_cls_token_id).count() == 1
            && tokens.iter().filter(|&&x| x == c.img_end_id).count() == 1,
        "exactly one image per request is supported"
    );
    Ok((start, end))
}

pub(crate) fn positions(tokens: &[u32], patch_hw: &[[f32; 2]], c: &ModelConfig) -> Result<(Vec<usize>, Vec<[f32; 2]>)> {
    let mut temporal = Vec::with_capacity(tokens.len());
    let mut spatial = Vec::with_capacity(tokens.len());
    let mut count = 0usize;
    let mut patch = 0;
    for &token in tokens {
        if ![
            c.img_id,
            c.image_reg_1_token_id,
            c.image_reg_2_token_id,
            c.image_reg_3_token_id,
            c.image_reg_4_token_id,
            c.img_end_id,
        ]
        .contains(&token)
        {
            count += 1;
        }
        ensure!(count > 0, "image continuation before any class/text token");
        temporal.push(count - 1);
        spatial.push(if token == c.img_id {
            let p = *patch_hw.get(patch).context("missing patch coordinates")?;
            patch += 1;
            p
        } else {
            [f32::NAN, f32::NAN]
        });
    }
    if patch != patch_hw.len() {
        bail!("unused patch coordinates");
    }
    Ok((temporal, spatial))
}

#[cfg(test)]
mod tests {
    use safetensors::{Dtype, SafeTensors};

    use super::*;
    fn config() -> ModelConfig {
        serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap()
    }

    #[test]
    fn image_positions_preserve_registers_and_resume_text() {
        let c = config();
        let tokens = [244, 245, 246, 247, 248, 227, 227, 230, 524, 257];
        let (t, hw) = positions(&tokens, &[[-1., -0.5], [1., 0.5]], &c).unwrap();
        assert_eq!(t, [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]);
        assert!(hw[..5].iter().flatten().all(|v| v.is_nan()));
        assert_eq!(hw[5], [-1., -0.5]);
        assert_eq!(image_range(&tokens, &c).unwrap(), (0, 7));
        assert!(positions(&[227], &[[0., 0.]], &c).is_err());
        assert!(image_range(&[244, 230, 244, 230], &c).is_err());
    }

    #[test]
    fn spatial_rotary_makes_paired_prefix_keys_distinct() {
        let c = config();
        let temporal = temporal_factors(&c);
        let golden = (0..c.n_heads * c.head_dim / 2)
            .map(|i| i as f32 / 256.)
            .collect::<Vec<_>>();
        let mut rope = Vec::new();
        rotary_factors(
            &c,
            &[0, 16383],
            &[[0.2, 0.7], [f32::NAN; 2]],
            &golden,
            &temporal,
            &mut rope,
        );
        let pairs = c.head_dim / 2;
        assert_ne!(&rope[..pairs], &rope[pairs..2 * pairs]);
        for h in 1..c.n_heads {
            assert_eq!(
                &rope[c.n_heads * pairs..(c.n_heads + 1) * pairs],
                &rope[(c.n_heads + h) * pairs..(c.n_heads + h + 1) * pairs]
            );
        }
    }

    #[test]
    #[ignore = "diagnostic requires independently exported GPU RoPE operators"]
    fn report_rotary_operator_differences() {
        let bytes = std::fs::read("artifacts/reference/rope-operators.safetensors").unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let f32s = |name: &str| -> Vec<f32> {
            let t = tensors.tensor(name).unwrap();
            assert_eq!(t.dtype(), Dtype::F32);
            t.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        let c = config();
        let temporal = temporal_factors(&c);
        let reference = f32s("freqs_cis_all");
        let delta = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            let mut max = 0f64;
            let mut squares = 0f64;
            let mut unequal = 0usize;
            for (&x, &y) in a.iter().zip(b) {
                if x.is_nan() && y.is_nan() {
                    continue;
                }
                let d = (x as f64 - y as f64).abs();
                max = max.max(d);
                squares += d * d;
                unequal += usize::from(x.to_bits() != y.to_bits());
            }
            serde_json::json!({"max_abs":max,"rms_abs":(squares/a.len() as f64).sqrt(),"unequal":unequal})
        };
        let time = tensors.tensor("pos_t").unwrap();
        assert_eq!(time.dtype(), Dtype::I64);
        let time = time
            .data()
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()) as usize)
            .collect::<Vec<_>>();
        let hw = f32s("pos_hw").chunks_exact(2).map(|x| [x[0], x[1]]).collect::<Vec<_>>();
        let golden = f32s("golden_freqs");
        let mut rope = Vec::new();
        rotary_factors(&c, &time, &hw, &golden, &temporal, &mut rope);
        let mut report = serde_json::json!({"temporal_table":delta(bytemuck::cast_slice(&temporal),&reference)});
        for layer in [0, 6, 12, 21] {
            for kind in ["q", "k"] {
                let mut values = f32s(&format!("layer.{layer}.{kind}.input"));
                let expected = f32s(&format!("layer.{layer}.{kind}.expected"));
                assert_eq!(values.len() / 2, rope.len());
                for (v, &[cos, sin]) in values.chunks_exact_mut(2).zip(&rope) {
                    let a = v[0];
                    let b = v[1];
                    v[0] = a * cos - b * sin;
                    v[1] = a * sin + b * cos;
                }
                report[format!("layer.{layer}.{kind}")] = delta(&values, &expected);
            }
        }
        std::fs::write(
            "reference/rotary-rust-probe.json",
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
