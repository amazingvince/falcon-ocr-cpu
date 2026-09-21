use super::*;
use crate::kernels::{self, Simd};

fn supported() {
    assert!(Simd::Avx2.validate().is_ok(), "Selected tests require AVX2/FMA; no silent skip");
}

fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|i| {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        if i % 97 == 0 { -0.0 } else {
            (((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32) * scale
        }
    }).collect()
}

fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "coordinate {index}: {a:?} vs {b:?}");
    }
}

struct Case {
    q: Vec<f32>, prefix_k: Vec<f32>, generated_k: Vec<f32>, v: Vec<f32>, sinks: Vec<f32>,
    total: usize, prefix: usize, offset: usize, image_start: usize, image_end: usize,
    rows: usize, heads: usize, kv_heads: usize, width: usize,
}

impl Case {
    #[allow(clippy::too_many_arguments)]
    fn new(total: usize, prefix: usize, offset: usize, image_start: usize,
           image_end: usize, rows: usize, heads: usize, kv_heads: usize, width: usize) -> Self {
        let qw = heads * width;
        let kw = kv_heads * width;
        Self {
            q: data(rows * qw, 41, 0.75),
            prefix_k: data(prefix * qw, 53, 0.75),
            generated_k: data((total - prefix) * kw, 67, 0.75),
            v: data(total * kw, 79, 1.0),
            sinks: (0..heads).map(|h| match h % 4 {
                0 => -1000.0, 1 => 1000.0, 2 => -0.0, _ => 2.25,
            }).collect(),
            total, prefix, offset, image_start, image_end, rows, heads, kv_heads, width,
        }
    }

    fn compare(&self, simd: Simd, threads: usize) {
        let mut actual = vec![f32::NAN; self.q.len()];
        let mut expected = actual.clone();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            kernels::attention_compact_combined_for_staged_test(
                &self.q, &self.prefix_k, &self.generated_k, &self.v, self.rows,
                self.prefix, self.total, self.heads, self.kv_heads, self.width,
                self.offset, self.image_start, self.image_end, &self.sinks,
                &mut expected, simd);
            kernels::attention_compact_with_simd(
                &self.q, &self.prefix_k, &self.generated_k, &self.v, self.rows,
                self.prefix, self.total, self.heads, self.kv_heads, self.width,
                self.offset, self.image_start, self.image_end, &self.sinks,
                &mut actual, simd);
        });
        assert!(actual.iter().all(|v| v.is_finite()));
        exact(&actual, &expected);
    }

    // Construct exact, head-dependent first-coordinate dot products. Each
    // successive tile has a larger maximum, forcing a real rounded rescale.
    fn set_increasing_tile_maxima(&mut self) {
        assert_eq!(self.width, 64);
        self.q.fill(0.0);
        for h in 0..self.heads { self.q[h * 64] = 1.0; }
        self.prefix_k.fill(0.0);
        self.generated_k.fill(0.0);
        for key in 0..self.total {
            let score = (key / 128) as f32 * 1.25 + (key % 17) as f32 * 0.03125;
            if key < self.prefix {
                for h in 0..self.heads {
                    self.prefix_k[(key * self.heads + h) * 64] = 8.0 * (score + h as f32 * 0.0625);
                }
            } else {
                for h in 0..self.kv_heads {
                    self.generated_k[((key - self.prefix) * self.kv_heads + h) * 64] =
                        8.0 * (score + h as f32 * 0.0625);
                }
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn compare_pv(len: usize, rescale: f32) {
    // Deliberately offset all views; gaps between value heads must be ignored.
    let stride = 8 * 64;
    let first = 3 * 64;
    let values = data(1 + len * stride, 107, 1e10);
    let values = &values[1..];
    let probabilities: Vec<f32> = (0..len).map(|j| match j % 7 {
        0 => 0.0, 1 => -0.0, 2 => f32::from_bits(1), 3 => 0.125,
        4 => 1.0, 5 => 0.999_999_94, _ => 0.03125,
    }).collect();
    let guard = f32::from_bits(0x4a12_3456);
    let initial = data(64, 137, 1e10);
    let mut actual = vec![guard; 70];
    actual[3..67].copy_from_slice(&initial);
    let mut expected = initial;
    for value in &mut expected { *value *= rescale; }
    unsafe {
        for (j, probability) in probabilities.iter().enumerate() {
            let begin = first + j * stride;
            kernels::x86::axpy_avx2(*probability, &values[begin..begin + 64], &mut expected);
        }
        pv_tile(&probabilities, values, first, stride, rescale, &mut actual[3..67]);
    }
    exact(&actual[3..67], &expected);
    assert!(actual[..3].iter().chain(&actual[67..]).all(|x| x.to_bits() == guard.to_bits()));
}

#[test]
fn pv_tile_preserves_rescale_fma_order_and_unaligned_guards() {
    supported();
    for len in [1, 2, 17, 63, 64, 127, 128] {
        for rescale in [0.0, -0.0, f32::from_bits(0x0080_0000), 0.25, 0.999_999_94, 1.0] {
            unsafe { compare_pv(len, rescale); }
        }
    }
}

#[test]
fn staged_tiles_preserve_increasing_maxima_and_cancellation() {
    supported();
    for total in [127, 128, 129, 255, 256, 257, 385] {
        let mut case = Case::new(total, total.min(129), total - 1, 1, total.min(128), 1, 16, 8, 64);
        case.set_increasing_tile_maxima();
        for (i, value) in case.v.iter_mut().enumerate() {
            let key = i / 512;
            *value = match key % 4 { 0 => 1e8, 1 => 1.0, 2 => -1e8, _ => -0.03125 };
            if i % 64 >= 32 { *value = -*value; }
        }
        case.compare(Simd::Avx2, 4);
    }
}

#[test]
fn negative_values_signed_zeros_and_sink_extremes_are_exact() {
    supported();
    for prefix in [0, 128, 257] {
        let image_start = usize::from(prefix != 0);
        let mut case = Case::new(257, prefix, 256, image_start, prefix.min(128), 1, 16, 8, 64);
        for (i, value) in case.v.iter_mut().enumerate() {
            *value = if i % 3 == 0 { -0.0 } else { -value.abs() * 1e8 };
        }
        case.compare(Simd::Avx2, 1);
        case.v.fill(-0.0);
        case.compare(Simd::Avx2, 4);
    }
}

#[test]
fn paired_heads_masks_and_prefix_crossovers_are_exact() {
    supported();
    for kv_heads in [1, 4, 8, 16] {
        for prefix in [0, 127, 128, 129, 257] {
            let image_start = usize::from(prefix != 0);
            Case::new(257, prefix, 256, image_start, prefix.min(129), 1, 16, kv_heads, 64)
                .compare(Simd::Avx2, 4);
        }
    }
    for offset in [0, 1, 126, 127, 128, 129, 256] {
        Case::new(257, 129, offset, 1, 129, 1, 16, 8, 64).compare(Simd::Avx2, 4);
    }
    Case::new(257, 129, 256, 1, 129, 1, 16, 8, 64).compare(Simd::Auto, 1);
}

#[test]
fn real_prefix_full_context_and_partial_final_tiles_are_exact() {
    supported();
    for (total, prefix) in [(6544, 6544), (6545, 6544), (16383, 6544), (16384, 6544)] {
        Case::new(total, prefix, total - 1, 1, 6540, 1, 16, 8, 64).compare(Simd::Avx2, 4);
    }
}

#[test]
fn public_validation_and_unchanged_paths_are_preserved() {
    supported();
    for width in [31, 63, 65, 80] {
        Case::new(17, 9, 16, 1, 8, 1, 16, 8, width).compare(Simd::Avx2, 4);
    }
    for rows in [0, 2, 3, 4, 17] {
        Case::new(17, 9, 17 - rows, 1, 8, rows, 16, 8, 64).compare(Simd::Avx2, 4);
    }
    for simd in [Simd::Scalar, Simd::Avx512] {
        if simd.validate().is_ok() {
            Case::new(17, 9, 16, 1, 8, 1, 16, 8, 64).compare(simd, 4);
        }
    }
    for invalid in 0..3 {
        let mut case = Case::new(17, 9, 16, 1, 8, 1, 16, 8, 64);
        match invalid {
            0 => { case.generated_k.pop(); },
            1 => case.image_end = case.total + 1,
            _ => case.offset = case.total,
        }
        let mut output = vec![f32::from_bits(0x4a12_3456); 1024];
        let before = output.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kernels::attention_compact_with_simd(
                &case.q, &case.prefix_k, &case.generated_k, &case.v, case.rows,
                case.prefix, case.total, case.heads, case.kv_heads, case.width,
                case.offset, case.image_start, case.image_end, &case.sinks,
                &mut output, Simd::Avx2);
        }));
        assert!(result.is_err());
        exact(&output, &before);
    }
}
