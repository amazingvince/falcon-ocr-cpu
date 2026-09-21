//! Explicit BF16 graph operations. These are experimental building blocks;
//! they do not select a model precision or establish full-model GPU parity.
use half::bf16;
use rayon::prelude::*;

/// Reusable FP32 opmath buffers; subsequent same-sized calls need no allocation.
#[derive(Default)]
pub struct NormWorkspace {
    input: Vec<f32>,
    output: Vec<f32>,
    weight: Vec<f32>,
}

/// Match the pinned Torch contract: FP32 RMS opmath, optional affine multiply
/// still in FP32, then one BF16 output rounding. Epsilon is explicit: input and
/// Q/K norms use f32::EPSILON, and the learned final norm uses 1e-5.
pub fn rms_norm(
    input: &[bf16],
    output: &mut [bf16],
    width: usize,
    epsilon: f32,
    weight: Option<&[bf16]>,
    scratch: &mut NormWorkspace,
) {
    assert!(
        width > 0 && input.len().is_multiple_of(width),
        "BF16 RMSNorm input shape"
    );
    assert_eq!(input.len(), output.len(), "BF16 RMSNorm output shape");
    if let Some(weight) = weight {
        assert_eq!(weight.len(), width, "BF16 RMSNorm affine shape");
    }
    scratch.input.resize(input.len(), 0.0);
    scratch.output.resize(input.len(), 0.0);
    scratch
        .input
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(dst, src)| *dst = src.to_f32());
    let promoted_weight = weight.map(|weight| {
        scratch.weight.resize(width, 0.0);
        for (dst, src) in scratch.weight.iter_mut().zip(weight) {
            *dst = src.to_f32();
        }
        scratch.weight.as_slice()
    });
    crate::kernels::rms_norm(
        &scratch.input,
        &mut scratch.output,
        width,
        epsilon,
        promoted_weight,
    );
    output
        .par_iter_mut()
        .zip(scratch.output.par_iter())
        .for_each(|(dst, &src)| *dst = bf16::from_f32(src));
}

/// Interleaved gate/up projection. The upstream Triton kernel rounds the square
/// to BF16 before multiplying by up, then rounds that product to BF16 too.
pub fn squared_relu_gate(packed: &[bf16], output: &mut [bf16]) {
    assert_eq!(
        packed.len(),
        output
            .len()
            .checked_mul(2)
            .expect("BF16 gate shape overflow")
    );
    output
        .par_iter_mut()
        .zip(packed.par_chunks_exact(2))
        .for_each(|(dst, pair)| {
            let gate = pair[0].to_f32();
            let relu = if gate > 0.0 { gate } else { 0.0 };
            let square = bf16::from_f32(relu * relu).to_f32();
            *dst = bf16::from_f32(square * pair[1].to_f32());
        });
}

/// BF16 residual addition has a visible rounding boundary after every add.
pub fn residual_add(hidden: &mut [bf16], projected: &[bf16]) {
    assert_eq!(hidden.len(), projected.len(), "BF16 residual shape");
    hidden
        .par_iter_mut()
        .zip(projected.par_iter())
        .for_each(|(dst, src)| *dst = bf16::from_f32(dst.to_f32() + src.to_f32()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_gate_and_residual_rounding_are_explicit() {
        // This pair lies across the single-final-round versus two-round boundary.
        let mut witness = None;
        for gate_bits in 0x3f80..0x4080 {
            let gate = bf16::from_bits(gate_bits);
            let up = bf16::from_f32(1.1015625);
            let staged =
                bf16::from_f32(bf16::from_f32(gate.to_f32().powi(2)).to_f32() * up.to_f32());
            let once = bf16::from_f32(gate.to_f32().powi(2) * up.to_f32());
            if staged != once {
                witness = Some((gate, up, staged));
                break;
            }
        }
        let (gate, up, expected) = witness.expect("fixture must distinguish rounding contracts");
        let packed = [gate, up, bf16::from_f32(-3.0), up];
        let mut out = [bf16::ZERO; 2];
        squared_relu_gate(&packed, &mut out);
        assert_eq!(out, [expected, bf16::ZERO]);
        let mut hidden = [bf16::ONE, bf16::from_f32(-4.0)];
        residual_add(
            &mut hidden,
            &[bf16::from_f32(0.00390625), bf16::from_f32(2.0)],
        );
        assert_eq!(hidden, [bf16::ONE, bf16::from_f32(-2.0)]); // even tie at 1.0
    }

    #[test]
    fn rms_uses_fp32_epsilon() {
        let input = [bf16::from_f32(0.00001); 64];
        let mut output = [bf16::ZERO; 64];
        rms_norm(
            &input,
            &mut output,
            64,
            f32::EPSILON,
            None,
            &mut NormWorkspace::default(),
        );
        let x = input[0].to_f32() as f64;
        let expected = bf16::from_f64(x / (x * x + f32::EPSILON as f64).sqrt());
        assert!(output.iter().all(|&v| v == expected));
        let wrong_epsilon = bf16::from_f64(x / (x * x + bf16::EPSILON.to_f32() as f64).sqrt());
        assert_ne!(output[0], wrong_epsilon);
    }
}
