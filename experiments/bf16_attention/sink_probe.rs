//! CPU-only replay of the production BF16 attention sink arithmetic.
//! Input: finite F32 bit patterns for raw BF16-promoted value, LSE, and sink.
use std::io::{self, BufRead};

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let bits: Vec<u32> = line
            .split_whitespace()
            .map(|s| u32::from_str_radix(s, 16).unwrap())
            .collect();
        assert_eq!(bits.len(), 3);
        let [raw, lse, sink] = std::array::from_fn(|i| f32::from_bits(bits[i]));
        assert!(raw.is_finite() && lse.is_finite() && sink.is_finite());
        assert_eq!(bits[0] & 0xffff, 0, "raw must be BF16-promoted");
        let scale = (1. + (sink - lse).exp()).recip();
        let pre_cast = raw * scale;
        assert!(scale.is_finite() && pre_cast.is_finite());
        let output = pre_cast.to_bits();
        let rounded = output.wrapping_add(0x7fff + ((output >> 16) & 1)) >> 16;
        println!("{:08x} {:08x} {rounded:04x}", scale.to_bits(), output);
    }
}
