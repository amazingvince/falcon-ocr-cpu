use super::*;
use super::super::{attention_original_for_test, attention_with_simd, Simd};

fn supported() {
    assert!(Simd::Avx2.validate().is_ok(), "These selected tests require AVX2/FMA; no silent skip");
}
fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|i| {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        if i % 97 == 0 { -0.0 } else {
            (((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32) * scale
        }
    }).collect()
}
fn exact(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "bit difference at {i}: {x:?} vs {y:?}");
    }
}

#[target_feature(enable="avx2,fma")]
unsafe fn compare_vectors(a: &[f32], b: &[f32], factor: f32) {
    unsafe {
        assert_eq!(dot64(a,b).to_bits(), super::super::x86::dot_avx2(a,b).to_bits());
        let mut actual = a.to_vec(); let mut expected = a.to_vec();
        axpy64(factor,b,&mut actual);
        super::super::x86::axpy_avx2(factor,b,&mut expected);
        exact(&actual,&expected);
    }
}

#[test]
fn vectors_preserve_four_accumulators_and_fma() {
    supported();
    for seed in 1..=128 {
        let a = data(64,seed,if seed%2==0 {1e10} else {0.03125});
        let b = data(64,seed+997,if seed%3==0 {1e-10} else {8.0});
        for factor in [-4.0,-0.0,0.0,0.125,3.25] {
            unsafe { compare_vectors(&a,&b,factor); }
        }
    }
    let a:Vec<_>=(0..64).map(|i| if i%2==0 {1.0} else {-1.0}).collect();
    unsafe { compare_vectors(&a,&[1.0;64],-0.0); }
}

fn compare_case(kv:usize, offset:usize, image_start:usize, image_end:usize,
                query_len:usize, width:usize, simd:Simd, extreme:bool, threads:usize) {
    let heads=16;
    let q=data(query_len*heads*width,11,if extreme {100.0} else {0.75});
    let k=data(kv*heads*width,23,if extreme {100.0} else {0.75});
    let v=data(kv*heads*width,37,if extreme {1e8} else {1.0});
    let sinks:Vec<_>=(0..heads).map(|h| match h%4 {0=>-1000.0,1=>1000.0,2=>0.0,_=>2.25}).collect();
    let mut actual=vec![f32::NAN;q.len()]; let mut expected=actual.clone();
    let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        attention_original_for_test(&q,&k,&v,query_len,kv,heads,width,offset,image_start,image_end,
                                    &sinks,&mut expected,simd);
        attention_with_simd(&q,&k,&v,query_len,kv,heads,width,offset,image_start,image_end,
                            &sinks,&mut actual,simd);
    });
    assert!(actual.iter().all(|x|x.is_finite()));
    exact(&actual,&expected);
}

#[test]
fn causal_tile_tails_are_bit_exact() {
    supported();
    for kv in [1,2,17,63,64,127,128,129,143,144,161,255,256,257,1025] {
        compare_case(kv,kv-1,0,0,1,64,Simd::Avx2,false,4);
    }
}
#[test]
fn image_and_causal_boundaries_are_bit_exact() {
    supported();
    for (kv,offset,start,end) in [(17,0,1,16),(17,1,1,16),(17,15,1,16),(17,16,1,16),
        (161,0,1,129),(161,1,1,129),(161,128,1,129),(161,129,1,129),(161,160,1,129)] {
        compare_case(kv,offset,start,end,1,64,Simd::Avx2,false,4);
    }
}
#[test]
fn extreme_logits_sinks_and_long_context_are_bit_exact() {
    supported();
    compare_case(257,256,1,144,1,64,Simd::Avx2,true,4);
    // Real full-page prefix and full context endpoint; no timing assertion.
    for kv in [6544,16384] { compare_case(kv,kv-1,1,6540,1,64,Simd::Avx2,false,4); }
}
#[test]
fn auto_and_one_thread_are_bit_exact() {
    supported();
    compare_case(161,160,1,129,1,64,Simd::Auto,false,1);
}
#[test]
fn other_shapes_backends_and_prefill_stay_on_original_path() {
    supported();
    for simd in [Simd::Scalar,Simd::Avx512] {
        if simd.validate().is_ok() { compare_case(17,16,1,16,1,64,simd,false,4); }
    }
    for width in [31,63,65,80] { compare_case(17,16,1,16,1,width,Simd::Avx2,false,4); }
    for rows in [2,3,4,17] { compare_case(17,17-rows,1,16,rows,64,Simd::Avx2,false,4); }
}
