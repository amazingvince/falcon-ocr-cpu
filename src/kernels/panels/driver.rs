//! The row-block / panel-split scheduler and the tile epilogue shared by the
//! FP32 and BF16 panel GEMMs. Scheduling never changes arithmetic: every
//! output is one FMA chain over ascending `k` inside the micro-kernel, so
//! results are independent of the thread count and the split.
use rayon::prelude::*;

use super::panel::Epilogue;

/// Rows per micro-kernel call.
pub(super) const MR: usize = 6;

/// A panel micro-kernel over `NR` output channels: it packs a row block of
/// `A` into its operand layout and computes `MR x NR` tiles against its
/// weight panels.
pub(super) trait PanelKernel<const NR: usize>: Sync {
    /// Element type of a packed row block (`padded x k` elements).
    type Packed: Copy + Default + Send;
    /// Pack `rows` rows of `a` (`rows x k`, row `r` times `row_scale[r]`) as
    /// `padded` rows (zero rows past `rows`) into `out`.
    fn pack_rows(
        &self,
        a: &[f32],
        rows: usize,
        padded: usize,
        k: usize,
        row_scale: Option<&[f32]>,
        out: &mut [Self::Packed],
    );
    /// One `MR x NR` tile of packed row group `a` (`MR x k`) against weight
    /// panel `panel`.
    ///
    /// # Safety
    /// The kernel's instruction set is available (the caller checked its
    /// `available`).
    unsafe fn tile(&self, k: usize, a: &[Self::Packed], panel: usize, tile: &mut [[f32; NR]; MR]);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Store,
    Glu,
    Add,
}

/// `C = A * W^T` for `a` (`m x k`, row-major) through `kernel`, written via
/// `epilogue`. With `row_scale`, row `r` of `A` is used as `a[r][k] *
/// row_scale[r]` (an RMS norm folded into packing). Row blocks are sized so
/// the packed block stays in L2 (about 1 MB on Zen 4); the panel range is
/// split so every thread has work.
pub(super) fn tiled_gemm<const NR: usize, P: PanelKernel<NR>>(
    kernel: &P,
    a: &[f32],
    m: usize,
    k: usize,
    n: usize,
    row_scale: Option<&[f32]>,
    epilogue: Epilogue<'_>,
) {
    assert_eq!(a.len(), m * k, "panel GEMM input shape");
    assert_eq!(n % NR, 0, "panel GEMM needs whole panels");
    if let Some(scale) = row_scale {
        assert_eq!(scale.len(), m, "panel GEMM row scales");
    }
    let (out, mode) = match epilogue {
        Epilogue::Store(out) => (out, Mode::Store),
        Epilogue::Glu(out) => (out, Mode::Glu),
        Epilogue::Add(out) => (out, Mode::Add),
    };
    let out_width = if mode == Mode::Glu { n / 2 } else { n };
    assert_eq!(out.len(), m * out_width, "panel GEMM output shape");
    if m == 0 || n == 0 {
        return;
    }
    let block_rows = if k <= 1024 { 96 } else { 48 };
    let row_blocks = m.div_ceil(block_rows);
    let panels_total = n / NR;
    let threads = rayon::current_num_threads().max(1);
    let splits = (2 * threads).div_ceil(row_blocks).clamp(1, panels_total);
    let per_split = panels_total.div_ceil(splits);
    let shared = crate::team::SharedMut::new(out);
    let store = |row: usize, panel: usize, values: &[f32; NR]| match mode {
        Mode::Glu => {
            // SAFETY: (row, panel) tiles are disjoint across units.
            let dst = unsafe { shared.slice(row * out_width + panel * (NR / 2), NR / 2) };
            for (i, y) in dst.iter_mut().enumerate() {
                *y = crate::kernels::squared_relu_glu(values[2 * i], values[2 * i + 1]);
            }
        }
        Mode::Add => {
            // SAFETY: as above.
            let dst = unsafe { shared.slice(row * out_width + panel * NR, NR) };
            for (y, v) in dst.iter_mut().zip(values) {
                *y += v;
            }
        }
        Mode::Store => {
            // SAFETY: as above.
            unsafe { shared.slice(row * out_width + panel * NR, NR) }.copy_from_slice(values);
        }
    };
    (0..row_blocks * splits).into_par_iter().for_each_init(
        || vec![P::Packed::default(); block_rows.next_multiple_of(MR) * k],
        |packed, unit| {
            let (block, split) = (unit / splits, unit % splits);
            let row0 = block * block_rows;
            let rows = block_rows.min(m - row0);
            let padded = rows.next_multiple_of(MR);
            kernel.pack_rows(
                &a[row0 * k..(row0 + rows) * k],
                rows,
                padded,
                k,
                row_scale.map(|s| &s[row0..row0 + rows]),
                packed,
            );
            let first = split * per_split;
            let last = (first + per_split).min(panels_total);
            let mut tile = [[0.0_f32; NR]; MR];
            for panel in first..last {
                for g in 0..padded / MR {
                    // SAFETY: the caller checked the kernel's `available`;
                    // the packed group holds `MR x k` values.
                    unsafe { kernel.tile(k, &packed[g * MR * k..(g + 1) * MR * k], panel, &mut tile) };
                    let valid = MR.min(rows - g * MR);
                    for (r, values) in tile[..valid].iter().enumerate() {
                        store(row0 + g * MR + r, panel, values);
                    }
                }
            }
        },
    );
}
