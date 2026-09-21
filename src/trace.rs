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
