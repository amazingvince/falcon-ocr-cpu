#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn compact(
    q: &[f32], prefix_k: &[f32], generated_k: &[f32], v: &[f32], prefix_len: usize,
    n_heads: usize, n_kv_heads: usize, query_offset: usize,
    image_start: usize, image_end: usize, sinks: &[f32], output: &mut [f32],
) {
    let scale = (64_f32).sqrt().recip();
    output.par_chunks_mut(64).enumerate().for_each(|(qh, out)| {
        // SAFETY: The parent validated features, slices and shape relationships.
        unsafe { compact_head(qh,q,prefix_k,generated_k,v,prefix_len,n_heads,n_kv_heads,
                              query_offset,image_start,image_end,sinks,scale,out); }
    });
}

#[target_feature(enable="avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn compact_head(
    qh: usize, q: &[f32], prefix_k: &[f32], generated_k: &[f32], v: &[f32],
    prefix_len: usize, n_heads: usize, n_kv_heads: usize, query_offset: usize,
    image_start: usize, image_end: usize, sinks: &[f32], scale: f32, out: &mut [f32],
) {
    let head_dim=64;
    let query_width=n_heads*head_dim;
    let kv_width=n_kv_heads*head_dim;
    let repeat=n_heads/n_kv_heads;
    // SAFETY: The original safe entry point checked shapes/features. This body
    // is extracted verbatim except its direct fixed64 dot and AXPY call targets.
    unsafe {
        // GENERATED_ORIGINAL_COMPACT_HEAD_BODY
    }
}
