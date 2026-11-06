//! One layer of the quantized prefill on pools of 1, 2, 3 and 7 threads.
//!
//! The page pipeline prefills a page on a second pool whose size differs
//! from the runner's, and its tokens are a sequential run's only because
//! every prefill stage computes each output the same way whatever its pool's
//! thread count, while the partitions (row blocks, panel splits, records)
//! change with it. This runs the product's stages of one layer, at a page's
//! row count (at least `PARALLEL_ROWS`, so the row-parallel and fused passes
//! run), for W8 with FP32 and INT16 panels and for W16, and compares
//! every output bit for bit: the QKV projection with the RMS norm folded
//! into row scales, the rotary factors, the fused QKV pass with its BF16
//! attention copies, prefill attention (FP32 with either softmax, and
//! BF16), the output projection and FFN (residual adds and the gate in the
//! epilogues), sealing in every split format (the rotated research ones
//! too), and the unfused split with its residual add.
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
    config::{CacheLayout, ExpMode, ModelConfig, PrefillSoftmax, Tuning},
    kernels::{self, Bf16Kv, Geometry, PrefillOptions, PrefillProjection as Projection, Simd},
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
    // One layer is enough to compare sealing across pool sizes.
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
        ..PrefillOptions::EXACT
    };
    // BF16 attention copies where the CPU has AVX512-BF16, and the INT16
    // panels where the plan has them.
    let bf16 = kernels::prefill_bf16_rows_available();
    let i16 = kernels::prefill_plan(Some(8), false, Simd::Auto, &Tuning::default()).projection;
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
            false,
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
                let code_only = (i16 != Projection::PanelAvx2).then_some(i16);
                for kernel in std::iter::once(Projection::PanelAvx2).chain(code_only) {
                    if kernel != Projection::PanelAvx2 && *bits_per_code != 8 {
                        continue;
                    }
                    let body = match kernel {
                        Projection::PanelAvx2 => format!("W{bits_per_code}"),
                        _ => format!("W{bits_per_code} {kernel}"),
                    };
                    let mut push = |stage: &str, value: Vec<u32>| out.push((format!("{body}: {stage}"), value));
                    let mut work = Workspace::default();
                    work.resize(rows, &c);
                    let mut h = hidden.clone();
                    panel_qkv(&c, layer, &h, rows, &mut work, kernel);
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
                        for softmax in [PrefillSoftmax::Online, PrefillSoftmax::Static] {
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
                                PrefillOptions { softmax, ..fast },
                            );
                            let stage = match softmax {
                                PrefillSoftmax::Online => "BF16 attention",
                                PrefillSoftmax::Static => "BF16 attention, static softmax",
                            };
                            push(stage, bits(&attention));
                        }
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
                    let mut attention = vec![f32::NAN; rows * qdim];
                    let fp32_static = PrefillOptions {
                        fp32_softmax: PrefillSoftmax::Static,
                        ..PrefillOptions::EXACT
                    };
                    cache.layers[0].attention(
                        &work.q,
                        rows,
                        rows,
                        &c,
                        geometry,
                        &sinks,
                        &mut attention,
                        Simd::Auto,
                        None,
                        fp32_static,
                    );
                    push("attention, static softmax", bits(&attention));
                    panel_out_ffn(
                        &c,
                        layer,
                        &mut h,
                        rows,
                        &mut work,
                        kernel,
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
                        super::cache::seal_layer(
                            &mut sealed.layers,
                            0,
                            rows,
                            sealed.capacity,
                            &c,
                            mode,
                            ExpMode::Fast,
                            &sealed.tuning,
                        )
                        .unwrap();
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
    // The BF16 and INT16 stages ran where the CPU has them: their outputs
    // are not the FP32 ones.
    let output = |stage: &str| &single.iter().find(|(name, _)| name == stage).expect(stage).1;
    if i16 != Projection::PanelAvx2 {
        assert_ne!(
            output("W8: QKV projection"),
            output(&format!("W8 {i16}: QKV projection"))
        );
    }
    // Near-exact's default: the static tiles wherever the CPU has them.
    let planned = kernels::prefill_plan(Some(16), false, Simd::Auto, &Tuning::default()).attention;
    if planned.fp32_softmax() == PrefillSoftmax::Static {
        assert_ne!(output("W16: attention"), output("W16: attention, static softmax"));
    }
    if bf16 {
        assert_ne!(output("W8: attention"), output("W8: BF16 attention"));
        assert_ne!(
            output("W8: BF16 attention"),
            output("W8: BF16 attention, static softmax")
        );
    }
    for threads in [2, 3, 7] {
        let other = run(threads);
        assert_eq!(other.len(), single.len());
        for ((stage, one), (_, many)) in single.iter().zip(&other) {
            assert!(one == many, "{stage}: {threads} threads differ from one");
        }
    }
}

/// `kv-layers` seals the chosen layers exactly as a Q8 cache does and the rest
/// as a Q16 cache does, with Q16's exp on all of them.
#[test]
fn kv_layers_seal_the_chosen_layers_at_q8() {
    let mut c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
    c.n_layers = 2;
    let rows = 16;
    let mut keys = values(rows * c.query_dim(), 1);
    // A GQA pair shares the temporal half of its key.
    for row in 0..rows {
        for group in 0..c.n_kv_heads {
            let start = (row * c.n_heads + 2 * group) * 64;
            keys.copy_within(start..start + 32, start + 64);
        }
    }
    let kv = values(rows * c.kv_dim(), 2);
    let seal = |mask: u32, layer: usize, mode: Kv| {
        let compact = || LayerCache::Compact {
            prefix_k: keys.clone(),
            generated_k: Vec::new(),
            v: kv.clone(),
            prefix_len: rows,
        };
        let mut layers = vec![compact(), compact()];
        let tuning = Tuning {
            kv_q8_layers: mask,
            ..Tuning::default()
        };
        super::cache::seal_layer(&mut layers, layer, rows, rows + 8, &c, mode, ExpMode::Fast, &tuning).unwrap();
        let LayerCache::Split(split) = &layers[layer] else {
            unreachable!("sealed");
        };
        (split.record_bits(), split.fast_exp())
    };
    let (q8, _) = seal(0, 1, Kv::Q8);
    let (q16, _) = seal(0, 0, Kv::Q16);
    assert_ne!(q8, q16);
    assert_eq!(seal(0b10, 1, Kv::Q16), (q8, false));
    assert_eq!(seal(0b10, 0, Kv::Q16), (q16, false));
    // A whole Q8 cache under `ExpMode::Fast` takes the polynomial exp.
    assert!(seal(0, 0, Kv::Q8).1);
}
