//! Test-only numerical interventions. Never included in production libraries.
use rayon::prelude::*;
use std::cell::Cell;

#[cfg(any(windows, target_os = "linux"))]
#[path = "../examples/support/aocl_dynamic.rs"]
mod aocl_dynamic;
#[cfg(any(windows, target_os = "linux"))]
thread_local! {
    static AOCL_PREFILL_W2: std::cell::RefCell<Option<aocl_dynamic::Aocl>> = const { std::cell::RefCell::new(None) };
}

/// Each diagnostic worker loads and retains its own verified library handle.
/// No mutable process-wide dispatch state is used; production builds omit this.
pub(crate) fn set_thread_aocl_prefill_w2(
    path: &std::path::Path,
    sha256: &str,
) -> anyhow::Result<()> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        let library = aocl_dynamic::Aocl::load(path, sha256)?;
        assert!(library.sha256().eq_ignore_ascii_case(sha256));
        AOCL_PREFILL_W2.with(|slot| *slot.borrow_mut() = Some(library));
        Ok(())
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    anyhow::bail!("AOCL diagnostic loader supports Windows and Linux only")
}

pub(crate) fn aocl_prefill_w2_override(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    channels: usize,
    output: &mut [f32],
) -> bool {
    // The pinned architecture has only W2 at [N=768,K=2304]. Restrict to
    // prefill-sized matrices; all layers use the same rule, without selected
    // layer indices or fixture-specific values. Decode remains unchanged.
    if rows <= 8 || width != 2304 || channels != 768 {
        return false;
    }
    #[cfg(any(windows, target_os = "linux"))]
    return AOCL_PREFILL_W2.with(|slot| {
        let state = slot.borrow();
        let Some(library) = state.as_ref() else {
            return false;
        };
        library
            .linear(input, rows, width, weight, channels, output, 1.0, 0.0)
            .expect("checked AOCL prefill W2 diagnostic call");
        true
    });
    #[cfg(not(any(windows, target_os = "linux")))]
    false
}

thread_local! { static RMS_VARIANT: Cell<u8> = const { Cell::new(0) }; }

/// Set only from an experimental Rayon pool's start handler. Each diagnostic
/// pool owns its worker threads, so concurrent tests and other pools stay unchanged.
pub(crate) fn set_thread_rms_variant(variant: u8) {
    assert!(variant <= 2);
    RMS_VARIANT.set(variant);
}

pub(crate) fn rms_override(
    input: &[f32],
    out: &mut [f32],
    width: usize,
    eps: f32,
    weight: Option<&[f32]>,
) -> bool {
    let variant = RMS_VARIANT.get();
    if variant == 0 || !matches!(width, 64 | 768) {
        return false;
    }
    out.par_chunks_mut(width)
        .zip(input.par_chunks(width))
        .for_each(|(target, source)| {
            let mut lanes = [0.0f32; 128];
            for (thread, sum) in lanes.iter_mut().enumerate() {
                for start in (thread * 4..width).step_by(512) {
                    for &x in &source[start..start + 4] {
                        *sum = x.mul_add(x, *sum);
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
            let variance = sum / width as f32 + eps;
            let scale = if variant == 2 {
                (1.0 / (variance as f64).sqrt()) as f32
            } else {
                variance.sqrt().recip()
            };
            for (index, (value, &x)) in target.iter_mut().zip(source).enumerate() {
                *value = x * scale;
                if let Some(weight) = weight {
                    *value *= weight[index];
                }
            }
        });
    true
}
