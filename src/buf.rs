//! Read-only typed buffers that are either owned or a range of a shared file
//! mapping, so kernel-ready model files are used in place (zero-copy).
use std::{ops::Range, sync::Arc};

use anyhow::{Result, ensure};
use memmap2::Mmap;

pub(crate) enum Buf<T: bytemuck::Pod> {
    Owned(Vec<T>),
    /// `range` (bytes) of `map`, checked for size and alignment at creation.
    Mapped {
        map: Arc<Mmap>,
        range: Range<usize>,
    },
}

impl<T: bytemuck::Pod> Buf<T> {
    /// A typed view of `range` in `map`; `len` elements expected.
    pub(crate) fn mapped(map: &Arc<Mmap>, range: Range<usize>, len: usize) -> Result<Self> {
        ensure!(range.end <= map.len(), "mapped tensor out of bounds");
        let typed: &[T] = bytemuck::try_cast_slice(&map[range.clone()])
            .map_err(|e| anyhow::anyhow!("mapped tensor size/alignment: {e}"))?;
        ensure!(typed.len() == len, "mapped tensor length {} != {len}", typed.len());
        Ok(Self::Mapped {
            map: Arc::clone(map),
            range,
        })
    }

    /// The raw bytes, for writing kernel-ready files.
    pub(crate) fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
}

impl<T: bytemuck::Pod> std::ops::Deref for Buf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        match self {
            Buf::Owned(values) => values,
            // Size and alignment were checked in `Buf::mapped`.
            Buf::Mapped { map, range } => bytemuck::cast_slice(&map[range.clone()]),
        }
    }
}

impl<T: bytemuck::Pod> std::fmt::Debug for Buf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Buf::Owned(v) => write!(f, "Owned({} elements)", v.len()),
            Buf::Mapped { range, .. } => write!(f, "Mapped({range:?})"),
        }
    }
}
