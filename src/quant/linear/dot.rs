//! Dot-product kernels of the quantized layer: the single- and multi-row
//! decode dots per code type and instruction set, and their selection.

type DotQ<C> = fn(&[f32], &[C], &[f32]) -> f32;
/// `out[r] = dot(x[r * stride..], codes, scales)` for `rows <= 8`, bitwise
/// the single-row [`DotQ`] per row, decoding the weights once.
type DotRowsQ<C> = fn(&[f32], usize, usize, &[C], &[f32], &mut [f32; 8]);

/// A multi-row dot kernel, where the backend has one.
#[derive(Clone, Copy)]
pub(super) enum DotRows {
    I8(DotRowsQ<i8>),
    I16(DotRowsQ<i16>),
    None,
}

/// The single-row dot kernel, selected once per matrix.
#[derive(Clone, Copy)]
pub(super) enum Dot {
    I8(DotQ<i8>),
    I16(DotQ<i16>),
}

/// `output[k] = fl(codes[k] * scales[k / group])`.
pub(super) fn fill_row<C: crate::simd::QCode>(codes: &[C], scales: &[f32], group: usize, output: &mut [f32]) {
    for (k, (value, &code)) in output.iter_mut().zip(codes).enumerate() {
        *value = code.to_f32() * scales[k / group];
    }
}

/// The dot kernel for code type `C`, `group` and the selected backend.
pub(super) fn select_dot<C: crate::simd::QCode>(group: usize, simd: crate::kernels::Simd) -> DotQ<C> {
    let selected = simd.resolved();
    #[cfg(target_arch = "x86_64")]
    if selected != crate::kernels::Simd::Scalar
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        return match group {
            // SAFETY (all): AVX2/FMA detected above.
            32 => |x, c, s| unsafe { dot_q_avx2::<C, 32>(x, c, s) },
            64 => |x, c, s| unsafe { dot_q_avx2::<C, 64>(x, c, s) },
            _ => |x, c, s| unsafe { dot_q_avx2::<C, 128>(x, c, s) },
        };
    }
    #[cfg(target_arch = "aarch64")]
    if selected == crate::kernels::Simd::Neon {
        return match group {
            // SAFETY (all): NEON is baseline on aarch64.
            32 => |x, c, s| unsafe { crate::simd::dot_q::<crate::simd::Neon, C, 32>(x, c, s) },
            64 => |x, c, s| unsafe { crate::simd::dot_q::<crate::simd::Neon, C, 64>(x, c, s) },
            _ => |x, c, s| unsafe { crate::simd::dot_q::<crate::simd::Neon, C, 128>(x, c, s) },
        };
    }
    let _ = selected;
    match group {
        32 => dot_q_scalar::<C, 32>,
        64 => dot_q_scalar::<C, 64>,
        _ => dot_q_scalar::<C, 128>,
    }
}

/// The multi-row dot for code type `C`, `group` and the selected backend
/// (vector backends only; the scalar path keeps per-row dots).
pub(super) fn select_dot_rows<C: crate::simd::QCode>(group: usize, simd: crate::kernels::Simd) -> Option<DotRowsQ<C>> {
    let selected = simd.resolved();
    #[cfg(target_arch = "x86_64")]
    if selected != crate::kernels::Simd::Scalar
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        return Some(match group {
            // SAFETY (all): AVX2/FMA detected above.
            32 => |x, st, n, c, s, o| unsafe { dot_rows_avx2::<C, 32>(x, st, n, c, s, o) },
            64 => |x, st, n, c, s, o| unsafe { dot_rows_avx2::<C, 64>(x, st, n, c, s, o) },
            _ => |x, st, n, c, s, o| unsafe { dot_rows_avx2::<C, 128>(x, st, n, c, s, o) },
        });
    }
    #[cfg(target_arch = "aarch64")]
    if selected == crate::kernels::Simd::Neon {
        return Some(match group {
            // SAFETY (all): NEON is baseline on aarch64.
            32 => |x, st, n, c, s, o| unsafe { dot_rows::<crate::simd::Neon, C, 32>(x, st, n, c, s, o) },
            64 => |x, st, n, c, s, o| unsafe { dot_rows::<crate::simd::Neon, C, 64>(x, st, n, c, s, o) },
            _ => |x, st, n, c, s, o| unsafe { dot_rows::<crate::simd::Neon, C, 128>(x, st, n, c, s, o) },
        });
    }
    let _ = (group, selected);
    None
}

/// Rows in groups of up to three (12 accumulators fit AVX2's registers).
#[inline(always)]
unsafe fn dot_rows<S: crate::simd::Simd, C: crate::simd::QCode, const G: usize>(
    x: &[f32],
    stride: usize,
    rows: usize,
    codes: &[C],
    scales: &[f32],
    out: &mut [f32; 8],
) {
    use crate::simd::dot_q_rows;
    debug_assert!(rows <= 8);
    let mut r = 0;
    unsafe {
        while r < rows {
            let x = &x[r * stride..];
            match rows - r {
                1 => {
                    out[r] = dot_q_rows::<S, C, G, 1>(x, stride, codes, scales)[0];
                    r += 1;
                }
                2 => {
                    out[r..r + 2].copy_from_slice(&dot_q_rows::<S, C, G, 2>(x, stride, codes, scales));
                    r += 2;
                }
                _ => {
                    out[r..r + 3].copy_from_slice(&dot_q_rows::<S, C, G, 3>(x, stride, codes, scales));
                    r += 3;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_rows_avx2<C: crate::simd::QCode, const G: usize>(
    x: &[f32],
    stride: usize,
    rows: usize,
    codes: &[C],
    scales: &[f32],
    out: &mut [f32; 8],
) {
    unsafe { dot_rows::<crate::simd::Avx2, C, G>(x, stride, rows, codes, scales, out) }
}

/// About two equal channel blocks per pool thread (multiples of 4 channels):
/// small decode matrices finish in one or two rounds without idle threads.
pub(super) fn balanced_tasks(channels: usize) -> usize {
    let target = 2 * crate::team::threads();
    let per = channels.div_ceil(target).next_multiple_of(4).max(4);
    channels.div_ceil(per)
}

/// Output base pointer shared by tasks that write disjoint elements.
#[derive(Clone, Copy)]
pub(super) struct OutputPtr(pub(super) *mut f32);
// SAFETY: tasks write disjoint (row, channel) elements of one exclusive borrow.
unsafe impl Send for OutputPtr {}
unsafe impl Sync for OutputPtr {}
impl OutputPtr {
    /// The base pointer.
    pub(super) fn get(self) -> *mut f32 {
        self.0
    }
}

/// Address of an exclusive output buffer for disjoint writes from tasks.
pub(super) fn output_ptr(output: &OutputCell) -> *mut f32 {
    output.0
}
/// `Sync` wrapper so the write closure can be shared across tasks.
pub(super) struct OutputCell(pub(super) *mut f32);
// SAFETY: see OutputPtr.
unsafe impl Send for OutputCell {}
unsafe impl Sync for OutputCell {}

/// `kernels::dot_scalar` over `fl(code * scale)` weights, same iterator sum.
fn dot_q_scalar<C: crate::simd::QCode, const G: usize>(x: &[f32], codes: &[C], scales: &[f32]) -> f32 {
    x.iter()
        .zip(codes)
        .enumerate()
        .map(|(k, (value, &code))| value * (code.to_f32() * scales[k / G]))
        .sum()
}

/// `kernels::x86::dot_avx2` over `fl(code * scale)` weights: the generic
/// `simd::dot_q` (same phase accumulators, tree and tails) under AVX2/FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q_avx2<C: crate::simd::QCode, const G: usize>(x: &[f32], codes: &[C], scales: &[f32]) -> f32 {
    unsafe { crate::simd::dot_q::<crate::simd::Avx2, C, G>(x, codes, scales) }
}
