//! Optional tensor capture for reproducible cross-runtime comparisons.
use anyhow::Result;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::{collections::BTreeMap, path::Path};

pub trait Trace: Send {
    fn enabled(&self) -> bool {
        true
    }
    /// Phase boundaries also fire when tensor capture is disabled, allowing
    /// allocation/profiling instruments to observe the steady decode loop.
    fn decode_start(&mut self) {}
    fn decode_end(&mut self) {}
    fn decode_step(&mut self, _rows: usize, _ms: f64, _kv_bytes: usize) {}
    fn prefix_sealed(&mut self, _before: usize, _after: usize, _ms: f64) {}
    fn cache_retired(&mut self, _bytes: usize) {}
    /// One screened vocabulary-head selection: rows recomputed exactly, and
    /// whether the step fell back to the full FP32 head.
    fn head_screen(&mut self, _candidates: usize, _fallback: bool) {}
    /// Whether teacher-forced runs should also compute the model's own greedy
    /// choice at every step and report it through [`Trace::teacher_step`].
    fn scores_teacher(&self) -> bool {
        false
    }
    /// Teacher-forced step `step`: the forced token and the token greedy
    /// decoding would have selected from the same prefix.
    fn teacher_step(&mut self, _step: usize, _forced: u32, _predicted: u32) {}
    /// Full next-token logits of teacher-forced step `step` (only when
    /// [`Trace::scores_teacher`] is true and the step evaluated the full head).
    fn teacher_logits(&mut self, _step: usize, _logits: &[f32]) {}
    /// Whether single-request forward passes should report every projection
    /// input through [`Trace::linear_input`] (quantization calibration).
    fn captures_linear_inputs(&self) -> bool {
        false
    }
    /// Row-major input `[rows, width]` of projection `site` (`qkv`, `wo`,
    /// `w13` or `w2`) in `layer`.
    fn linear_input(&mut self, _layer: usize, _site: &'static str, _rows: usize, _data: &[f32]) {}
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()>;
}
pub struct NoTrace;
impl Trace for NoTrace {
    #[inline]
    fn enabled(&self) -> bool {
        false
    }
    #[inline]
    fn tensor(&mut self, _name: &str, _shape: &[usize], _data: &[f32]) -> Result<()> {
        Ok(())
    }
}

/// Give a single-request execution its batch request identity without changing
/// the model's canonical trace names or its profiling phase boundaries.
pub(crate) struct PrefixedTrace<'a> {
    inner: &'a mut dyn Trace,
    prefix: &'a str,
}
impl<'a> PrefixedTrace<'a> {
    pub(crate) fn new(inner: &'a mut dyn Trace, prefix: &'a str) -> Self {
        Self { inner, prefix }
    }
}
impl Trace for PrefixedTrace<'_> {
    fn enabled(&self) -> bool {
        self.inner.enabled()
    }
    fn decode_start(&mut self) {
        self.inner.decode_start();
    }
    fn decode_end(&mut self) {
        self.inner.decode_end();
    }
    fn decode_step(&mut self, rows: usize, ms: f64, kv_bytes: usize) {
        self.inner.decode_step(rows, ms, kv_bytes);
    }
    fn prefix_sealed(&mut self, before: usize, after: usize, ms: f64) {
        self.inner.prefix_sealed(before, after, ms);
    }
    fn cache_retired(&mut self, bytes: usize) {
        self.inner.cache_retired(bytes);
    }
    fn head_screen(&mut self, candidates: usize, fallback: bool) {
        self.inner.head_screen(candidates, fallback);
    }
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        self.inner
            .tensor(&format!("{}.{name}", self.prefix), shape, data)
    }
}
#[derive(Default)]
pub struct TensorTrace {
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}
impl Trace for TensorTrace {
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        self.tensors
            .insert(name.into(), (shape.to_vec(), data.to_vec()));
        Ok(())
    }
}
impl TensorTrace {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let views = self
            .tensors
            .iter()
            .map(|(name, (shape, data))| {
                Ok((
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytemuck::cast_slice(data))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, path.as_ref())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_adapter_preserves_disabled_phase_callbacks() {
        #[derive(Default)]
        struct Probe {
            starts: usize,
            ends: usize,
        }
        impl Trace for Probe {
            fn enabled(&self) -> bool {
                false
            }
            fn decode_start(&mut self) {
                self.starts += 1;
            }
            fn decode_end(&mut self) {
                self.ends += 1;
            }
            fn tensor(&mut self, _: &str, _: &[usize], _: &[f32]) -> Result<()> {
                panic!("disabled tracing must not capture tensors")
            }
        }
        let mut probe = Probe::default();
        let mut trace = PrefixedTrace::new(&mut probe, "request.2");
        assert!(!trace.enabled());
        trace.decode_start();
        trace.decode_end();
        assert_eq!((probe.starts, probe.ends), (1, 1));
    }
}
