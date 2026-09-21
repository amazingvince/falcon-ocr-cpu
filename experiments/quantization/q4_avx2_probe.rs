//! Standalone diagnostic: rustc --edition 2024 [--test] q4_avx2_probe.rs.
//! Synthetic operator evidence only; no model quality or performance claims.
mod q4_avx2;
mod q4_reference;

use q4_avx2::{Backend, linear};
use q4_reference::Q4Linear;

const WIDTHS: &[usize] = &[
    1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 768, 1024, 2304,
];
const CHANNELS: usize = 7;

fn original_weights(k: usize) -> Vec<f32> {
    (0..CHANNELS * k)
        .map(|i| ((i * 67 % 199) as f32 - 99.0) / 31.0)
        .collect()
}

fn activations(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| ((i * 101 % 257) as f32 - 128.0) / 37.0)
        .collect()
}

#[cfg(not(test))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 || args[0] != "--output" {
        return Err("usage: q4_avx2_probe --output NEW_JSON_PATH".into());
    }
    if !q4_avx2::avx2_available() {
        return Err("explicit AVX2/FMA operator unavailable on this host".into());
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    let mut out = std::io::BufWriter::new(file);
    writeln!(
        out,
        "{{\"schema\":1,\"scope\":\"synthetic operator only\",\"timing\":null,\"model_integration\":false,\"quality_qualification\":false,\"avx2_fma_available\":true,\"compiled_source_inventory_sha256\":\"{}\",\"cases\":[",
        option_env!("FOCR_AVX2_SOURCE_SHA256").unwrap_or("uncaptured")
    )?;
    let mut first = true;
    for &k in WIDTHS {
        let weights = original_weights(k);
        for group in [64, 128] {
            if !first {
                writeln!(out, ",")?;
            }
            first = false;
            let quant = Q4Linear::quantize(&weights, CHANNELS, k, group)?;
            write!(
                out,
                "{{\"k\":{k},\"n\":{CHANNELS},\"group_size\":{group},\"original_weights\":{weights:?},\"packed_codes\":{:?},\"scales\":{:?},\"batches\":[",
                quant.packed_codes(),
                quant.scales()
            )?;
            for (index, rows) in [1, 2, 4, 8].into_iter().enumerate() {
                if index != 0 {
                    write!(out, ",")?;
                }
                let input = activations(rows, k);
                let mut scalar = vec![0.0; rows * CHANNELS];
                let mut avx2 = vec![0.0; rows * CHANNELS];
                linear(&quant, &input, rows, &mut scalar, Backend::Scalar)?;
                linear(&quant, &input, rows, &mut avx2, Backend::Avx2)?;
                write!(
                    out,
                    "{{\"rows\":{rows},\"input\":{input:?},\"scalar\":{scalar:?},\"avx2\":{avx2:?}}}"
                )?;
            }
            write!(out, "]}}")?;
        }
    }
    writeln!(out, "]}}")?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct CountingAllocator;
    thread_local! {
        static TRACK: Cell<bool> = const { Cell::new(false) };
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }
    fn count() {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
    }
    // SAFETY: all operations forward unchanged to System; only this thread's
    // allocation count is inspected. TLS cells have constant initialization.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count();
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count();
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            count();
            unsafe { System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    /// Independently unpacks bytes; never calls the scalar dequantizer/operator.
    fn reconstructed(q: &Q4Linear, channel: usize, column: usize) -> f32 {
        let (_, k) = q.dimensions();
        let byte = q.packed_codes()[channel * k.div_ceil(2) + column / 2];
        let nibble = ((byte as u32 >> (4 * (column % 2))) & 15) as i32;
        let signed = if nibble < 8 { nibble } else { nibble - 16 };
        let scale = q.scales()[channel * k.div_ceil(q.group_size()) + column / q.group_size()];
        signed as f32 * scale
    }

    fn check_oracle(q: &Q4Linear, x: &[f32], rows: usize, y: &[f32]) {
        let (n, k) = q.dimensions();
        let ku = k as f64 * 2f64.powi(-24);
        for row in 0..rows {
            for channel in 0..n {
                let mut sum = 0.0f64;
                let mut absolute = 0.0f64;
                for column in 0..k {
                    let product =
                        x[row * k + column] as f64 * reconstructed(q, channel, column) as f64;
                    sum += product;
                    absolute += product.abs();
                }
                let error = (y[row * n + channel] as f64 - sum).abs();
                let bound = ku / (1.0 - ku) * absolute;
                assert!(
                    error <= bound,
                    "K={k} rows={rows} row={row} channel={channel}: {error} > frozen gamma_K bound {bound}"
                );
            }
        }
    }

    #[test]
    fn all_tail_boundaries_real_widths_and_batches_match_independent_fp64() {
        if !q4_avx2::avx2_available() {
            return;
        }
        for &k in WIDTHS {
            for group in [64, 128] {
                let q = Q4Linear::quantize(&original_weights(k), CHANNELS, k, group).unwrap();
                for rows in [1, 2, 4, 8] {
                    // Slice offsets deliberately avoid assuming SIMD alignment.
                    let mut x = vec![4321.0];
                    x.extend(activations(rows, k));
                    x.push(-4321.0);
                    let original = x.clone();
                    for backend in [Backend::Scalar, Backend::Avx2] {
                        let mut y = vec![9876.0; rows * CHANNELS + 2];
                        linear(
                            &q,
                            &x[1..x.len() - 1],
                            rows,
                            &mut y[1..rows * CHANNELS + 1],
                            backend,
                        )
                        .unwrap();
                        assert_eq!(y[0], 9876.0);
                        assert_eq!(y[rows * CHANNELS + 1], 9876.0);
                        assert_eq!(x, original);
                        check_oracle(&q, &x[1..x.len() - 1], rows, &y[1..rows * CHANNELS + 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn exact_known_nibbles_zero_groups_and_subnormal_dequantization() {
        if !q4_avx2::avx2_available() {
            return;
        }
        for group in [64, 128] {
            let mut weights = vec![0.0; group * 2 + 1];
            weights[..8].copy_from_slice(&[14.0, -14.0, 1.0, 3.0, 5.0, -1.0, -3.0, -5.0]);
            weights[2 * group] = f32::from_bits(1);
            let q = Q4Linear::quantize(&weights, 1, weights.len(), group).unwrap();
            assert_eq!(&q.packed_codes()[..4], &[0x97, 0x20, 0x02, 0xee]);
            assert_eq!(q.scales(), &[2.0, 0.0, f32::from_bits(1)]);
            for column in [0, 1, 2, 3, 4, 5, 6, 7, group, 2 * group] {
                let mut x = vec![0.0; weights.len()];
                x[column] = 1.0;
                let mut output = [0.0];
                linear(&q, &x, 1, &mut output, Backend::Avx2).unwrap();
                assert_eq!(output[0].to_bits(), reconstructed(&q, 0, column).to_bits());
            }
        }
        // Subnormal scales inside a vector, not only the scalar tail.
        let q = Q4Linear::quantize(&[f32::from_bits(1); 17], 1, 17, 64).unwrap();
        let mut output = [0.0];
        linear(&q, &[1.0; 17], 1, &mut output, Backend::Avx2).unwrap();
        assert_eq!(output[0].to_bits(), 17);
    }

    #[test]
    fn checked_errors_precede_writes_and_overflow_is_reported() {
        let q = Q4Linear::quantize(&[1.0; 17], 1, 17, 64).unwrap();
        for backend in [Backend::Scalar, Backend::Avx2] {
            let mut y = [123.0];
            assert!(linear(&q, &[1.0; 17], 3, &mut y, backend).is_err());
            assert!(linear(&q, &[1.0; 17], usize::MAX, &mut y, backend).is_err());
            assert!(linear(&q, &[1.0; 16], 1, &mut y, backend).is_err());
            assert!(linear(&q, &[1.0; 17], 1, &mut [], backend).is_err());
            assert!(linear(&q, &[f32::NAN; 17], 1, &mut y, backend).is_err());
            assert!(linear(&q, &[f32::INFINITY; 17], 1, &mut y, backend).is_err());
            for group in [32, 256] {
                let other = Q4Linear::quantize(&[1.0; 17], 1, 17, group).unwrap();
                assert!(linear(&other, &[1.0; 17], 1, &mut y, backend).is_err());
            }
            assert_eq!(y, [123.0]);
        }
        if q4_avx2::avx2_available() {
            let q = Q4Linear::quantize(&[2.0; 17], 1, 17, 64).unwrap();
            assert_eq!(
                linear(&q, &[f32::MAX; 17], 1, &mut [0.0], Backend::Avx2),
                Err("nonfinite accumulation")
            );
        } else {
            assert_eq!(
                linear(&q, &[1.0; 17], 1, &mut [0.0], Backend::Avx2),
                Err("AVX2 and FMA are required for the explicit AVX2 backend")
            );
        }
    }

    #[test]
    fn zero_heap_allocations_for_every_live_row_count_and_group() {
        if !q4_avx2::avx2_available() {
            return;
        }
        for group in [64, 128] {
            for rows in [1, 2, 4, 8] {
                for k in [17, 768, 1024, 2304] {
                    let q = Q4Linear::quantize(&original_weights(k), CHANNELS, k, group).unwrap();
                    let x = activations(rows, k);
                    let mut y = vec![0.0; rows * CHANNELS];
                    for backend in [Backend::Scalar, Backend::Avx2] {
                        linear(&q, &x, rows, &mut y, backend).unwrap();
                        ALLOCATIONS.with(|n| n.set(0));
                        TRACK.with(|enabled| enabled.set(true));
                        let result = linear(&q, &x, rows, &mut y, backend);
                        TRACK.with(|enabled| enabled.set(false));
                        let allocations = ALLOCATIONS.with(Cell::get);
                        result.unwrap();
                        assert_eq!(allocations, 0);
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn altered_fp_environment_is_rejected_without_mutation() {
        let q = Q4Linear::quantize(&[1.0; 17], 1, 17, 64).unwrap();
        let mut original = 0u32;
        // SAFETY: SSE instructions read/write valid stack u32s; MXCSR changes
        // affect this test thread only and are restored before any assertion.
        unsafe {
            std::arch::asm!("stmxcsr [{ptr}]", ptr=in(reg) &mut original, options(nostack, preserves_flags));
        }
        for bits in [1 << 15, 1 << 6, 1 << 13, 2 << 13, 3 << 13] {
            let altered = original | bits;
            unsafe {
                std::arch::asm!("ldmxcsr [{ptr}]", ptr=in(reg) &altered, options(nostack, preserves_flags));
            }
            let mut y = [123.0];
            let result = linear(&q, &[1.0; 17], 1, &mut y, Backend::Scalar);
            unsafe {
                std::arch::asm!("ldmxcsr [{ptr}]", ptr=in(reg) &original, options(nostack, preserves_flags));
            }
            assert!(result.is_err());
            assert_eq!(y, [123.0]);
        }
    }
}
