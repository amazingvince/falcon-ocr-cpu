//! CPU-only diagnostic: exact F32 argument bits -> Rust exp2 and BF16 RN-even bits.
//! No attention implementation, QK candidate, GPU work, or production integration.
use std::io::{self, BufRead};

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let bits = u32::from_str_radix(line.trim(), 16).unwrap();
        let value = f32::from_bits(bits).exp2();
        // Swapping maxima between two saved traces can make a diagnostic
        // argument slightly positive; retain its value instead of clipping it.
        assert!(value.is_finite() && value >= 0.);
        let output = value.to_bits();
        // All outputs are nonnegative finite; standard BF16 round-nearest-even.
        let rounded = output.wrapping_add(0x7fff + ((output >> 16) & 1)) >> 16;
        println!("{bits:08x} {output:08x} {rounded:04x}");
    }
}
