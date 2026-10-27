//! Ignored timing probes of decode attention over the split cache.
use super::*;

/// Verification attention time per step by row count (22 layers of Q8
/// caches, 6,544 prefix positions, 12 threads): the cost of each extra
/// drafted row.
#[test]
#[ignore = "timing probe; run in release with --nocapture"]
fn verify_rows_probe() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let (p, layers, generated) = (6544, 22, 8);
    let k: Vec<f32> = (0..p * c.query_dim())
        .map(|i| {
            let (t, rest) = (i / c.query_dim(), i % c.query_dim());
            let (h, d) = (rest / 64, rest % 64);
            let identity = if d < 32 { h / 2 } else { h };
            ((t * 31 + identity * 17 + d * 7) % 101) as f32 / 151.0 - 0.3
        })
        .collect();
    let v: Vec<f32> = (0..p * c.kv_dim()).map(|i| (i % 73) as f32 / 97.0 - 0.2).collect();
    let tail: Vec<f32> = (0..c.kv_dim()).map(|i| (i % 29) as f32 / 41.0 - 0.3).collect();
    let sinks = vec![0.0_f32; c.n_heads];
    let mode = std::env::var("PROBE_MODE").map_or(Kv::Q8, |m| match m.as_str() {
        "q16" => Kv::Q16,
        "f32" => Kv::F32Split,
        "q8r" => Kv::Q8Rot,
        "q4r" => Kv::Q4Rot,
        _ => Kv::Q8,
    });
    let caches: Vec<SplitPrefix> = (0..layers)
        .map(|_| {
            let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 16, &c, mode, None).unwrap();
            for _ in 0..generated {
                cache.push_unique(&tail, &tail);
            }
            cache
        })
        .collect();
    let threads = std::env::var("PROBE_THREADS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(12);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    println!("{threads} threads");
    pool.install(|| {
        let mut base = 0.0;
        for rows in [1, 2, 3, 5, 8] {
            let q: Vec<f32> = (0..rows * c.query_dim())
                .map(|i| (i % 61) as f32 / 21.0 - 1.4)
                .collect();
            let mut out = vec![0.0; q.len()];
            for cache in &caches {
                cache.attention_decode_rows(&q, rows, &sinks, &mut out, Simd::Auto);
            }
            let rounds = 10;
            let t = std::time::Instant::now();
            for _ in 0..rounds {
                for cache in &caches {
                    cache.attention_decode_rows(&q, rows, &sinks, &mut out, Simd::Auto);
                }
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
            if rows == 1 {
                base = ms;
            }
            let extra = if rows > 1 { (ms - base) / (rows - 1) as f64 } else { 0.0 };
            println!("{mode:?} rows {rows}: {ms:.2} ms/step, {extra:.2} ms per extra row");
        }
    });
}

/// Decode attention time per token over 22 layers' worth of caches (so the
/// records stream from memory) by record format and thread count.
#[test]
#[ignore = "timing probe; run in release with --nocapture"]
fn decode_attention_bandwidth_probe() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let (p, layers) = (6544, 22);
    let k: Vec<f32> = (0..p * c.query_dim())
        .map(|i| {
            let (t, rest) = (i / c.query_dim(), i % c.query_dim());
            let (h, d) = (rest / 64, rest % 64);
            let identity = if d < 32 { h / 2 } else { h };
            ((t * 31 + identity * 17 + d * 7) % 101) as f32 / 151.0 - 0.3
        })
        .collect();
    let v: Vec<f32> = (0..p * c.kv_dim()).map(|i| (i % 73) as f32 / 97.0 - 0.2).collect();
    let q: Vec<f32> = (0..c.query_dim()).map(|i| (i % 61) as f32 / 21.0 - 1.4).collect();
    let sinks = vec![0.0_f32; c.n_heads];
    for mode in [Kv::Q8, Kv::Q8Rot, Kv::Q4Rot, Kv::Q16, Kv::F32Split] {
        let caches: Vec<SplitPrefix> = (0..layers)
            .map(|_| {
                SplitPrefix::from_compact(&k, &v, p, p + 8, &c, mode, None)
                    .unwrap()
                    .with_chunks(4)
            })
            .collect();
        let bytes: usize = caches.iter().map(|x| x.records.bytes()).sum();
        for threads in [4, 8, 12, 16] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut out = vec![0.0; c.query_dim()];
            pool.install(|| {
                for cache in &caches {
                    cache.attention_decode(&q, p, &sinks, &mut out, Simd::Auto);
                }
                let rounds = 10;
                let t = std::time::Instant::now();
                for _ in 0..rounds {
                    for cache in &caches {
                        cache.attention_decode(&q, p, &sinks, &mut out, Simd::Auto);
                    }
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
                println!(
                    "{mode:?} {threads:2} threads: {ms:.2} ms/token, {:.0} MB, {:.1} GB/s",
                    bytes as f64 / 1e6,
                    bytes as f64 / (ms * 1e6)
                );
            });
        }
    }
}
