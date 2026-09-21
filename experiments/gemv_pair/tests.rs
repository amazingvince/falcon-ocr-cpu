use super::*;
use super::super::{linear_original_for_gemv_pair_test, linear_with_simd, Simd};

const SHAPES: [(usize, usize); 5] = [(768, 2048), (1024, 768), (768, 4608),
    (2304, 768), (768, 65536)];

fn supported() {
    assert!(Simd::Avx2.validate().is_ok(), "Selected tests require AVX2/FMA; no silent skip");
}

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|i| {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        if i % 97 == 0 { -0.0 } else {
            // Non-power-of-two-scaled finite values exercise genuine rounding.
            ((state >> 40) as i32 - (1 << 23)) as f32 / 999983.0
        }
    }).collect()
}

fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {i}: {a:?} versus {b:?}");
    }
}

fn compare(rows: usize, k: usize, n: usize, simd: Simd, threads: usize) {
    let input = data(rows * k, 7919);
    let weight = data(n * k, 65537);
    let mut actual = vec![f32::NAN; rows * n];
    let mut expected = actual.clone();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        linear_original_for_gemv_pair_test(&input, rows, k, &weight, n, &mut expected, simd);
        linear_with_simd(&input, rows, k, &weight, n, &mut actual, simd);
    });
    assert!(actual.iter().all(|v| v.is_finite()));
    exact(&actual, &expected);
}

#[test]
fn all_five_actual_shapes_every_output_bit_matches() {
    supported();
    // Includes all 65,536 vocabulary outputs, not channel sampling. Each
    // matrix is released before allocating the next one; no model is loaded.
    for (k, n) in SHAPES { compare(1, k, n, Simd::Avx2, 4); }
}

#[test]
fn auto_and_single_thread_match_on_guarded_shape() {
    supported();
    compare(1, 768, 2048, Simd::Auto, 1);
}

#[target_feature(enable = "avx2,fma")]
unsafe fn compare_pair(input: &[f32], first: &[f32], second: &[f32]) {
    unsafe {
        let actual = dot_pair(input, first, second);
        let left = super::super::x86::dot_avx2(input, first);
        let right = super::super::x86::dot_avx2(input, second);
        exact(&[actual.0, actual.1], &[left, right]);
    }
}

#[test]
fn paired_reductions_preserve_cancellation_zero_and_scale_boundaries() {
    supported();
    for k in [768, 1024, 2304] {
        for seed in 1..=12 {
            let input = data(k, seed);
            let first = data(k, seed + 1009);
            let second = data(k, seed + 4099);
            unsafe { compare_pair(&input, &first, &second); }
        }
        let input: Vec<_> = (0..k).map(|i| match i % 8 {
            0 => 1.0e10, 1 => -1.0e10, 2 => 1.0e-10, 3 => -1.0e-10,
            4 => 0.0, 5 => -0.0, 6 => f32::MIN_POSITIVE, _ => -f32::MIN_POSITIVE,
        }).collect();
        let first: Vec<_> = (0..k).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let second: Vec<_> = (0..k).map(|i| if i % 2 == 0 { 0.125 } else { -8.0 }).collect();
        unsafe { compare_pair(&input, &first, &second); }
        for zero in [0.0, -0.0] {
            unsafe { compare_pair(&vec![zero; k], &first, &second); }
        }
    }
}

#[test]
fn unaligned_slices_and_channel_block_boundaries_match() {
    supported();
    for k in [768, 1024, 2304] {
        // Offset each read/output independently. Direct block checks cover
        // first/last pairs around the 32-channel scheduling boundary.
        let input = data(k + 1, 53);
        let weights = data(64 * k + 3, 97);
        let mut storage = vec![f32::from_bits(0x3f123456); 66];
        let marker = storage[0].to_bits();
        let actual = &mut storage[1..65];
        for block in 0..2 {
            unsafe { channel_block(&input[1..],
                &weights[3 + block * 32 * k..3 + (block + 1) * 32 * k],
                &mut actual[block * 32..(block + 1) * 32]); }
        }
        for j in 0..64 {
            let expected = unsafe { super::super::x86::dot_avx2(&input[1..], &weights[3 + j * k..3 + (j + 1) * k]) };
            assert_eq!(actual[j].to_bits(), expected.to_bits(), "K={k}, channel={j}");
        }
        assert_eq!(storage[0].to_bits(), marker);
        assert_eq!(storage[65].to_bits(), marker);
    }
}

#[test]
fn other_shapes_batches_and_backends_keep_original_outputs() {
    supported();
    for (k, n) in [(0, 17), (31, 33), (767, 2048), (769, 33), (768, 2047), (768, 2049)] {
        compare(1, k, n, Simd::Avx2, 4);
    }
    for rows in [2, 4, 8] { compare(rows, 768, 2048, Simd::Avx2, 4); }
    compare(9, 64, 32, Simd::Avx2, 4);
    for simd in [Simd::Scalar, Simd::Avx512] {
        if simd.validate().is_ok() { compare(1, 768, 2048, simd, 4); }
    }
}

#[test]
fn shape_guard_is_exact_and_public_validation_precedes_dispatch() {
    supported();
    for (k, n) in SHAPES { assert!(supported_shape(k, n)); }
    for (k, n) in [(768, 768), (1024, 2048), (2304, 65536), (768, 2047), (0, 0)] {
        assert!(!supported_shape(k, n));
    }
    let input = vec![0.0; 768];
    let weight = vec![0.0; 768 * 2048];
    for malformed in 0..3 {
        let mut out = vec![0.0; if malformed == 2 { 2047 } else { 2048 }];
        let x = if malformed == 0 { &input[..767] } else { &input[..] };
        let w = if malformed == 1 { &weight[..weight.len() - 1] } else { &weight[..] };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            linear_with_simd(x, 1, 768, w, 2048, &mut out, Simd::Avx2);
        })).is_err());
    }
}

fn sha(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn bound_map(root: &std::path::Path, relative: &str, expected: &str) -> memmap2::Mmap {
    let file = std::fs::File::open(root.join(relative)).expect("missing pinned operator input");
    // SAFETY: Read-only map remains alive throughout the borrowed views. The
    // capture owner must hold these frozen inputs unchanged and rehash them
    // after the process exits; this test also validates bytes before use.
    let map = unsafe { memmap2::Mmap::map(&file) }.expect("map pinned operator input");
    assert_eq!(sha(&map), expected, "changed operator input {relative}");
    map
}

fn tensor_floats<'a>(view: &'a safetensors::tensor::TensorView<'_>, shape: &[usize]) -> &'a [f32] {
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    assert_eq!(view.shape(), shape);
    bytemuck::try_cast_slice(view.data()).expect("aligned little-endian F32 tensor")
}

fn real_case(name: &str, input: &[f32], weight: &[f32], n: usize, weight_name: &str) {
    assert!(input.iter().all(|v| v.is_finite()));
    let mut actual = vec![f32::NAN; n];
    let mut expected = actual.clone();
    linear_original_for_gemv_pair_test(input, 1, input.len(), weight, n, &mut expected, Simd::Avx2);
    linear_with_simd(input, 1, input.len(), weight, n, &mut actual, Simd::Avx2);
    assert!(actual.iter().all(|v| v.is_finite()));
    exact(&actual, &expected);
    eprintln!("GEMV_PAIR_REAL_CASE {}", serde_json::json!({
        "input": name, "weight": weight_name, "input_width": input.len(),
        "output_width": n, "compared_outputs": n, "bit_mismatches": 0,
        "input_raw_sha256": sha(bytemuck::cast_slice(input)),
        "weight_raw_sha256": sha(bytemuck::cast_slice(weight)),
        "original_output_raw_sha256": sha(bytemuck::cast_slice(&expected)),
        "candidate_output_raw_sha256": sha(bytemuck::cast_slice(&actual)),
    }));
}

#[test]
fn pinned_real_cpu_operands_match_all_five_projections() {
    supported();
    assert!(cfg!(target_endian = "little"), "pinned F32 views require little endian");
    let root = std::path::PathBuf::from(std::env::var("FOCR_GEMV_PAIR_ROOT")
        .expect("capture must provide root for mandatory pinned real-operand test"));
    let config_bytes = std::fs::read(root.join("artifacts/model/config.json")).unwrap();
    assert_eq!(sha(&config_bytes), "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf");
    let config: crate::config::ModelConfig = serde_json::from_slice(&config_bytes).unwrap();
    config.validate().unwrap();
    assert_eq!(config.norm_eps.to_bits(), 1.0e-5_f32.to_bits());
    let weights_map = bound_map(&root, "artifacts/model/model.safetensors",
        "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16");
    let crossover_map = bound_map(&root, "artifacts/diagnostics/fp32-crossover-layer7-v1/rust/tensors.safetensors",
        "4972d70e6ec5c6c744745282141abb7768dbea6314970affa49603cd955e102d");
    let smoke_map = bound_map(&root, "artifacts/cpu/smoke-trace-sinks-pairwise.safetensors",
        "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309");
    let weights = safetensors::SafeTensors::deserialize(&weights_map).unwrap();
    let crossover = safetensors::SafeTensors::deserialize(&crossover_map).unwrap();
    let smoke = safetensors::SafeTensors::deserialize(&smoke_map).unwrap();
    let cases = [
        ("cpu_state.layer.7.attention_norm", "layers.7.attention.wqkv.weight", 768, 2048),
        ("cpu_state.layer.7.attention", "layers.7.attention.wo.weight", 1024, 768),
        ("cpu_state.layer.7.ffn_norm", "layers.7.feed_forward.w13.weight", 768, 4608),
        ("cpu_state.layer.7.gate", "layers.7.feed_forward.w2.weight", 2304, 768),
    ];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| {
        for (stage, weight_name, k, n) in cases {
            let input_view = crossover.tensor(stage).expect("missing exact CPU branch stage");
            let input = tensor_floats(&input_view, &[144, k]);
            let weight_view = weights.tensor(weight_name).unwrap();
            let weight = tensor_floats(&weight_view, &[n, k]);
            real_case(&format!("{stage}[112]"), &input[112 * k..113 * k], weight, n, weight_name);
        }
        // Operator-only reconstruction of the vocabulary input. No Model,
        // Runner, tokenizer, attention, cache or model pipeline is invoked.
        let hidden_view = smoke.tensor("decode.1.layer.21.hidden").expect("missing exact decode stage");
        let hidden = tensor_floats(&hidden_view, &[1, 768]);
        let norm_view = weights.tensor("norm.weight").unwrap();
        let norm = tensor_floats(&norm_view, &[768]);
        let mut normalized = vec![0.0; 768];
        super::super::rms_norm(hidden, &mut normalized, 768, config.norm_eps, Some(norm));
        let output_view = weights.tensor("output.weight").unwrap();
        let output_weight = tensor_floats(&output_view, &[65536, 768]);
        real_case("decode.1.layer.21.hidden+original_final_rms_norm", &normalized,
            output_weight, 65536, "output.weight");
        eprintln!("GEMV_PAIR_VOCAB_INPUT {}", serde_json::json!({
            "hidden_raw_sha256": sha(hidden_view.data()), "norm_raw_sha256": sha(norm_view.data()),
            "norm_epsilon_bits": config.norm_eps.to_bits(), "reconstructed_input_raw_sha256": sha(bytemuck::cast_slice(&normalized)),
            "operator_only": true,
        }));
    });
}
