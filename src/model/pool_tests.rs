//! One layer of the quantized prefill on pools of 1, 2, 3 and 7 threads.
//!
//! The page pipeline prefills a page on a second pool whose size differs
//! from the runner's, and its tokens are a sequential run's only because
//! every prefill stage computes each output the same way whatever its pool's
//! thread count, while the partitions (row blocks, panel splits, records)
//! change with it. This runs the product's stages of one layer, at a page's
//! row count (at least `PARALLEL_ROWS`, so the row-parallel and fused passes
//! run), for W8 with FP32 and BF16 panels and for W16, and compares every
//! output bit for bit: the QKV projection with the RMS norm folded into row
//! scales, the rotary factors, the fused QKV pass with its BF16 attention
//! copies, prefill attention (FP32 and BF16), the output projection and FFN
//! (residual adds and the gate in the epilogues), sealing in every split
//! format (the rotated research ones too), and the unfused split with its
//! residual add.
//! `kernels::tests::prefill_kernels_are_bitwise_independent_of_the_pool_size`
//! covers the FP32 projections, attention, norms and first-token heads.
use std::sync::Arc;

use super::{
    Layer, PARALLEL_ROWS, PhaseClock, Weight, add_residual,
    cache::{LayerCache, Session, Workspace},
    fused_prefix_rows, panel_out_ffn, panel_qkv,
    rope::temporal_factors,
    rotary_factors, split_norm_rope,
};
use crate::{
    config::{CacheLayout, ExpMode, ModelConfig, Tuning},
    kernels::{self, Bf16Kv, Geometry, PrefillOptions, Simd},
    quant::{Kv, linear::QuantLinear},
};

/// A fixed generator in [-1, 1), as in `kernels::tests`.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn weight(matrix: Option<QuantLinear>) -> Weight {
    Weight {
        range: 0..0,
        quantized: matrix.map(Arc::new),
    }
}

/// A layer of `bits`-bit matrices (group 64) with small random weights.
fn layer(c: &ModelConfig, bits: u32) -> Layer {
    let matrix = |out: usize, inputs: usize, seed: u32| {
        let w: Vec<f32> = values(out * inputs, seed).iter().map(|x| x * 0.05).collect();
        Some(QuantLinear::quantize_bits(&w, out, inputs, 64, bits).unwrap())
    };
    let (dim, qdim, kdim) = (c.dim, c.query_dim(), c.kv_dim());
    Layer {
        qkv: weight(matrix(qdim + 2 * kdim, dim, 11)),
        wo: weight(matrix(dim, qdim, 13)),
        w13: weight(matrix(2 * c.ffn_dim, dim, 17)),
        w2: weight(matrix(dim, c.ffn_dim, 19)),
        sinks: weight(None),
    }
}

#[test]
fn layer_prefill_is_bitwise_independent_of_the_pool_size() {
    if !kernels::panel_gemm::available(Simd::Auto) {
        return;
    }
    let mut c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
    // Sealing seals every layer of a session; one is enough here.
    c.n_layers = 1;
    let rows = PARALLEL_ROWS + 6;
    let (image_start, image_end) = (5, rows - 4);
    let (dim, qdim) = (c.dim, c.query_dim());
    // Text rows advance the temporal position; image rows carry spatial
    // coordinates (the registers and image end neither).
    let (mut t, mut hw) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for row in 0..rows {
        let image = (image_start + 1..image_end).contains(&row);
        t.push(if row < image_start {
            row
        } else {
            image_start + row.saturating_sub(image_end)
        });
        hw.push(if image {
            let i = (row - image_start - 1) as f32;
            [(i / 8.0).floor() / 4.0 - 1.0, (i % 8.0) / 4.0 - 1.0]
        } else {
            [f32::NAN; 2]
        });
    }
    let temporal = temporal_factors(&c);
    let golden = values(c.n_heads * c.head_dim / 2, 3);
    let hidden = values(rows * dim, 5);
    let sinks = values(c.n_heads, 7);
    let geometry = Geometry::new(0, image_start, image_end);
    let fast = PrefillOptions {
        exp: ExpMode::Fast,
        profile: false,
    };
    // BF16 panels and attention copies where the CPU has AVX512-BF16.
    let bf16 = kernels::prefill_bf16_rows_available();
    let bodies = [(8, layer(&c, 8)), (16, layer(&c, 16))];
    let session = || {
        Session::new(
            &c,
            rows + 8,
            rows,
            image_start,
            image_end,
            Simd::Auto,
            CacheLayout::Compact,
            ExpMode::Fast,
            Tuning::default(),
        )
        .unwrap()
    };
    let run = |threads: usize| -> Vec<(String, Vec<u32>)> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let mut out = Vec::new();
            for (bits_per_code, layer) in &bodies {
                for panels_bf16 in [false, true] {
                    if panels_bf16 && (*bits_per_code != 8 || !bf16) {
                        continue;
                    }
                    let body = format!("W{bits_per_code}{}", if panels_bf16 { " BF16 panels" } else { "" });
                    let mut push = |stage: &str, value: Vec<u32>| out.push((format!("{body}: {stage}"), value));
                    let mut work = Workspace::default();
                    work.resize(rows, &c);
                    let mut h = hidden.clone();
                    panel_qkv(&c, layer, &h, rows, &mut work, panels_bf16);
                    push("QKV projection", bits(&work.qkv));
                    rotary_factors(&c, &t, &hw, &golden, &temporal, &mut work.rope);
                    push(
                        "rotary factors",
                        work.rope.iter().flatten().map(|x| x.to_bits()).collect(),
                    );
                    let mut cache = session();
                    let copies = bf16.then_some((&mut work.bf16_keys, &mut work.bf16_values));
                    fused_prefix_rows(
                        &c,
                        rows,
                        &work.qkv,
                        &work.rope,
                        &mut work.q,
                        &mut cache.layers[0],
                        copies,
                        Simd::Auto,
                    );
                    let LayerCache::Compact { prefix_k, v, .. } = &cache.layers[0] else {
                        unreachable!("a compact session");
                    };
                    let (keys, values) = (prefix_k.clone(), v.clone());
                    push("fused queries", bits(&work.q));
                    push("fused keys", bits(&keys));
                    push("fused values", bits(&values));
                    if bf16 {
                        push("BF16 key copies", work.bf16_keys.clone());
                        push("BF16 value copies", work.bf16_values.clone());
                        let mut attention = vec![f32::NAN; rows * qdim];
                        let converted = Some(Bf16Kv::Converted(&work.bf16_keys, &work.bf16_values));
                        cache.layers[0].attention(
                            &work.q,
                            rows,
                            rows,
                            &c,
                            geometry,
                            &sinks,
                            &mut attention,
                            Simd::Auto,
                            converted,
                            fast,
                        );
                        push("BF16 attention", bits(&attention));
                    }
                    cache.layers[0].attention(
                        &work.q,
                        rows,
                        rows,
                        &c,
                        geometry,
                        &sinks,
                        &mut work.attn,
                        Simd::Auto,
                        None,
                        PrefillOptions::EXACT,
                    );
                    push("attention", bits(&work.attn));
                    panel_out_ffn(
                        &c,
                        layer,
                        &mut h,
                        rows,
                        &mut work,
                        panels_bf16,
                        &mut PhaseClock::new(rows, false),
                    );
                    push("gated FFN", bits(&work.gated));
                    push("layer output", bits(&h));
                    for mode in [Kv::F32Split, Kv::Q16, Kv::Q8, Kv::Q8Rot, Kv::Q4Rot] {
                        let mut sealed = session();
                        sealed.layers[0] = LayerCache::Compact {
                            prefix_k: keys.clone(),
                            generated_k: Vec::new(),
                            v: values.clone(),
                            prefix_len: rows,
                        };
                        sealed.len = rows;
                        sealed.seal_prefix(&c, mode, ExpMode::Fast).unwrap();
                        let LayerCache::Split(split) = &sealed.layers[0] else {
                            unreachable!("sealed");
                        };
                        push(&format!("sealed {mode:?}"), split.record_bits());
                    }
                }
            }
            // The unfused split (expanded caches, traced runs) and the
            // residual add of unpanelled bodies, row-parallel at this size.
            let mut work = Workspace::default();
            work.resize(rows, &c);
            work.qkv = values(work.qkv.len(), 23);
            rotary_factors(&c, &t, &hw, &golden, &temporal, &mut work.rope);
            split_norm_rope(&c, rows, &mut work, true);
            for (stage, value) in [
                ("split queries", &work.q),
                ("split keys", &work.k),
                ("split values", &work.v),
            ] {
                out.push((stage.to_owned(), bits(value)));
            }
            let mut h = hidden.clone();
            add_residual(&mut h, &values(rows * dim, 29), dim);
            out.push(("residual add".to_owned(), bits(&h)));
            out
        })
    };
    let single = run(1);
    // The BF16 stages ran where the CPU has them: their outputs are not
    // the FP32 ones.
    let output = |stage: &str| &single.iter().find(|(name, _)| name == stage).expect(stage).1;
    if bf16 {
        assert_ne!(output("W8: QKV projection"), output("W8 BF16 panels: QKV projection"));
        assert_ne!(output("W8: attention"), output("W8: BF16 attention"));
    }
    for threads in [2, 3, 7] {
        let other = run(threads);
        assert_eq!(other.len(), single.len());
        for ((stage, one), (_, many)) in single.iter().zip(&other) {
            assert!(one == many, "{stage}: {threads} threads differ from one");
        }
    }
}
