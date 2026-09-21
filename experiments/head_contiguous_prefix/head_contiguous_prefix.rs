//! Copied-project experiment only: reorder retained prefix K/V, preserve bits.
use crate::kernels::{self, Simd};
use anyhow::{Context, Result, ensure};

pub(crate) struct HeadContiguousPrefixCache {
    prefix_k: Vec<f32>,
    prefix_v: Vec<f32>,
    generated_k: Vec<f32>,
    generated_v: Vec<f32>,
    prefix: usize,
    capacity: usize,
    len: usize,
}

fn duplicate_heads_equal(expanded: &[f32]) -> bool {
    expanded.chunks_exact(128).all(|pair| {
        pair[..64]
            .iter()
            .zip(&pair[64..])
            .all(|(a, b)| a.to_bits() == b.to_bits())
    })
}

/// Reuse caller-reserved temporary token-major compact V for unchanged prefill.
/// Validate before clearing; no allocation and no arithmetic on stored values.
pub(crate) fn pack_prefill_values(expanded: &[f32], scratch: &mut Vec<f32>) -> Result<()> {
    ensure!(expanded.len().is_multiple_of(1024), "invalid expanded prefill V shape");
    ensure!(duplicate_heads_equal(expanded), "duplicated prefill V bits differ");
    ensure!(scratch.capacity() >= expanded.len() / 2, "prefill V scratch was not reserved");
    scratch.clear();
    for pair in expanded.chunks_exact(128) {
        scratch.extend_from_slice(&pair[..64]);
    }
    Ok(())
}

impl HeadContiguousPrefixCache {
    pub(crate) fn new(prefix: usize, capacity: usize, heads: usize, kv_heads: usize, dim: usize) -> Result<Self> {
        ensure!((heads, kv_heads, dim) == (16, 8, 64), "head-contiguous prefix requires 16Q/8KV/64");
        ensure!(prefix > 0 && prefix <= capacity, "invalid head-contiguous prefix/capacity");
        fn reserve(rows: usize, width: usize) -> Result<Vec<f32>> {
            let elements = rows.checked_mul(width).context("cache size overflow")?;
            // Explicit byte bound before asking the allocator for an address space.
            elements.checked_mul(std::mem::size_of::<f32>()).context("cache bytes overflow")?;
            let mut result = Vec::new();
            result.try_reserve_exact(elements).context("allocate head-contiguous cache")?;
            Ok(result)
        }
        Ok(Self {
            prefix_k: reserve(prefix, 1024)?,
            prefix_v: reserve(prefix, 512)?,
            generated_k: reserve(capacity - prefix, 512)?,
            generated_v: reserve(capacity - prefix, 512)?,
            prefix,
            capacity,
            len: 0,
        })
    }

    pub(crate) fn append(&mut self, k: &[f32], v: &[f32], offset: usize) -> Result<()> {
        ensure!(k.len() == v.len() && k.len().is_multiple_of(1024), "invalid expanded K/V shape");
        let rows = k.len() / 1024;
        ensure!(offset == self.len, "cache append is not contiguous");
        ensure!(offset.checked_add(rows).is_some_and(|end| end <= self.capacity), "cache capacity exceeded");
        let prefill = offset == 0;
        ensure!(if prefill { rows == self.prefix } else { offset >= self.prefix && rows == 1 },
            "only one complete prefix then single-token continuation supported");
        // Prefix K heads are intentionally independent after spatial RoPE.
        // All validation precedes any writes, including late invalid pairs.
        ensure!(duplicate_heads_equal(v), "duplicated V bits differ");
        ensure!(prefill || duplicate_heads_equal(k), "duplicated generated K bits differ");
        if prefill {
            for head in 0..16 {
                for token in 0..self.prefix {
                    let begin = (token * 16 + head) * 64;
                    self.prefix_k.extend_from_slice(&k[begin..begin + 64]);
                }
            }
            for head in 0..8 {
                for token in 0..self.prefix {
                    let begin = (token * 16 + head * 2) * 64;
                    self.prefix_v.extend_from_slice(&v[begin..begin + 64]);
                }
            }
        } else {
            for (kg, vg) in k.chunks_exact(128).zip(v.chunks_exact(128)) {
                self.generated_k.extend_from_slice(&kg[..64]);
                self.generated_v.extend_from_slice(&vg[..64]);
            }
        }
        self.len += rows;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &self, q: &[f32], current_expanded_k: &[f32], prefill_compact_v: &[f32],
        rows: usize, total_len: usize, offset: usize, image_start: usize, image_end: usize,
        sinks: &[f32], output: &mut [f32], simd: Simd,
    ) -> Result<()> {
        ensure!(self.len == total_len && offset.checked_add(rows) == Some(total_len), "attention/cache interval differs");
        if offset == 0 {
            ensure!(rows == self.prefix && self.generated_k.is_empty() && self.generated_v.is_empty(), "partial prefill unsupported");
            ensure!(current_expanded_k.len() == self.prefix * 1024, "prefill must borrow expanded K");
            ensure!(prefill_compact_v.len() == self.prefix * 512, "prefill must borrow token-major compact V");
            // Preserve the exact old prefill operator, input values and strides.
            kernels::attention_compact_with_simd(q, current_expanded_k, &[], prefill_compact_v,
                rows, self.prefix, total_len, 16, 8, 64, offset, image_start, image_end, sinks, output, simd);
        } else {
            ensure!(rows == 1 && offset >= self.prefix, "multiquery continuation unsupported");
            ensure!(prefill_compact_v.is_empty(), "decode must not retain prefill scratch contents");
            kernels::attention_head_contiguous_prefix_with_simd(q,
                &self.prefix_k, &self.prefix_v, &self.generated_k, &self.generated_v,
                rows, self.prefix, total_len, 16, 8, 64, offset, image_start, image_end, sinks, output, simd);
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    // GENERATED_STORAGE_TESTS
}
