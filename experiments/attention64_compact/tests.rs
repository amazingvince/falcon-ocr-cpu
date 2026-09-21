use super::super::{attention_compact_original_for_test,attention_compact_with_simd};

fn compact_case(total:usize,prefix:usize,offset:usize,image_start:usize,image_end:usize,
                rows:usize,heads:usize,kvheads:usize,width:usize,simd:Simd,extreme:bool) {
    let qw=heads*width;let kw=kvheads*width;let repeat=heads/kvheads;
    let q=data(rows*qw,41,if extreme {100.0} else {0.75});
    let pk=data(prefix*qw,53,if extreme {100.0} else {0.75});
    let gk=data((total-prefix)*kw,67,if extreme {100.0} else {0.75});
    let v=data(total*kw,79,if extreme {1e8} else {1.0});
    let sinks:Vec<_>=(0..heads).map(|h|match h%4 {0=>-1000.0,1=>1000.0,2=>0.0,_=>2.25}).collect();
    let mut expected=vec![f32::NAN;q.len()];let mut actual=expected.clone();let mut expanded_out=expected.clone();
    let mut expanded_k=vec![0.0;total*qw];let mut expanded_v=expanded_k.clone();
    for token in 0..total {for head in 0..heads {
        let target=token*qw+head*width;
        let key=if token<prefix {&pk[target..target+width]} else {
            let source=(token-prefix)*kw+(head/repeat)*width;&gk[source..source+width]
        };
        expanded_k[target..target+width].copy_from_slice(key);
        let source=token*kw+(head/repeat)*width;
        expanded_v[target..target+width].copy_from_slice(&v[source..source+width]);
    }}
    let pool=rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(||{
        attention_compact_original_for_test(&q,&pk,&gk,&v,rows,prefix,total,heads,kvheads,width,
            offset,image_start,image_end,&sinks,&mut expected,simd);
        attention_compact_with_simd(&q,&pk,&gk,&v,rows,prefix,total,heads,kvheads,width,
            offset,image_start,image_end,&sinks,&mut actual,simd);
        attention_original_for_test(&q,&expanded_k,&expanded_v,rows,total,heads,width,
            offset,image_start,image_end,&sinks,&mut expanded_out,simd);
    });
    assert!(actual.iter().all(|x|x.is_finite()));exact(&actual,&expected);exact(&actual,&expanded_out);
}

#[test]
fn compact_zero_full_and_tile_crossovers_are_bit_exact() {
    supported();
    for total in [1,17,127,128,129,257] {
        for prefix in [0,total/2,total] {
            compact_case(total,prefix,total-1,0,0,1,16,8,64,Simd::Avx2,false);
        }
    }
    for prefix in [127,128,129] {
        compact_case(257,prefix,256,1,prefix,1,16,8,64,Simd::Avx2,false);
    }
}
#[test]
fn compact_image_and_generated_boundaries_are_bit_exact() {
    supported();
    for offset in [0,1,126,127,128,129,160] {
        compact_case(161,129,offset,1,128,1,16,8,64,Simd::Avx2,false);
    }
}
#[test]
fn compact_real_prefix_full_context_and_extremes_are_bit_exact() {
    supported();
    for (total,prefix) in [(6544,6544),(6545,6544),(16384,6544)] {
        compact_case(total,prefix,total-1,1,6540,1,16,8,64,Simd::Avx2,false);
    }
    compact_case(257,129,256,1,129,1,16,8,64,Simd::Avx2,true);
}
#[test]
fn compact_gqa_repeats_and_auto_are_bit_exact() {
    supported();
    for kvheads in [4,8,16] {
        compact_case(257,129,256,1,128,1,16,kvheads,64,Simd::Auto,false);
    }
}
#[test]
fn compact_other_shapes_and_prefill_stay_unchanged() {
    supported();
    for width in [31,63,65,80] {
        compact_case(17,9,16,1,8,1,16,8,width,Simd::Avx2,false);
    }
    for rows in [2,3,4,17] {
        compact_case(17,9,17-rows,1,8,rows,16,8,64,Simd::Avx2,false);
    }
    for simd in [Simd::Scalar,Simd::Avx512] {
        if simd.validate().is_ok(){compact_case(17,9,16,1,8,1,16,8,64,simd,false);}
    }
}
