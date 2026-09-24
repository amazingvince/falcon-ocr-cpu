//! Diagnostic only: a hash-verified observed CUDA rsqrt table, never a backend.
//! Table entries correspond to ascending F32 input bits [0x3f800000,0x40800000).
pub const TABLE_LEN: usize = 1 << 24;
pub const MIN_ARGUMENT_BITS: u32 = 0x3400_0000; // f32::EPSILON = 2^-23
pub const MAX_ARGUMENT_BITS: u32 = 0x7f7f_ffff;

fn index_and_exponent_shift(bits: u32) -> (usize, i32) {
    assert!((MIN_ARGUMENT_BITS..=MAX_ARGUMENT_BITS).contains(&bits),
        "rsqrt table supports only positive normal F32 values >= f32::EPSILON");
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    let parity = exponent.rem_euclid(2);
    let index = ((parity as u32) << 23) | (bits & 0x007f_ffff);
    (index as usize, -(exponent - parity) / 2)
}

/// Return an observed-function reconstruction, with no allocation or file I/O.
///
/// The caller must bind the complete table to a successful exhaustive GPU
/// validation report. Panics on unsupported arguments, wrong length or an
/// invalid selected table entry. This function alone does not attest a table.
pub fn rsqrt_from_table(value: f32, table: &[f32]) -> f32 {
    assert_eq!(table.len(), TABLE_LEN, "canonical rsqrt table length differs");
    let (index, shift) = index_and_exponent_shift(value.to_bits());
    let canonical_bits = table[index].to_bits();
    assert!((0x3f00_0000..=0x3f80_0000).contains(&canonical_bits),
        "canonical rsqrt table entry is outside [0.5,1]");
    // Multiplication by 2^shift changes only the exponent. All results in the
    // supported domain remain positive normal; no subnormal rounding occurs.
    let result = i64::from(canonical_bits) + i64::from(shift) * (1_i64 << 23);
    assert!((0x0080_0000..=0x7f7f_ffff).contains(&result));
    f32::from_bits(result as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    const MANTISSAS: [u32; 7] = [0, 1, 0x1f_ffff, 0x3f_ffff, 0x40_0000, 0x7f_fffe, 0x7f_ffff];
    fn sparse_analytic_table() -> &'static [f32] {
        static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
        TABLE.get_or_init(|| {
            let mut table = vec![f32::NAN; TABLE_LEN];
            for parity in 0..2 {
                for mantissa in MANTISSAS {
                    let bits = 0x3f80_0000 + (parity << 23) + mantissa;
                    table[(bits - 0x3f80_0000) as usize] =
                        (1.0 / f64::from(f32::from_bits(bits)).sqrt()) as f32;
                }
            }
            table
        })
    }

    #[test]
    fn canonical_index_covers_every_entry_in_ascending_order() {
        for index in 0..TABLE_LEN {
            assert_eq!(index_and_exponent_shift(0x3f80_0000 + index as u32), (index, 0));
        }
    }

    #[test]
    fn every_supported_exponent_and_mantissa_boundaries_match_independent_f64() {
        let table = sparse_analytic_table();
        for exponent in 104..=254 {
            for mantissa in MANTISSAS {
                let value = f32::from_bits((exponent << 23) | mantissa);
                let expected = (1.0 / f64::from(value).sqrt()) as f32;
                assert_eq!(rsqrt_from_table(value, table).to_bits(), expected.to_bits(),
                    "input {:08x}", value.to_bits());
            }
        }
    }

    #[test]
    fn powers_of_four_rescale_exactly_in_both_directions() {
        let table = sparse_analytic_table();
        for power in -11..=63 {
            let value = (2.0_f64).powi(2 * power) as f32;
            let expected = (2.0_f64).powi(-power) as f32;
            assert_eq!(rsqrt_from_table(value, table).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn unsupported_arguments_are_rejected() {
        for bits in [0, 0x8000_0000, 1, 0x007f_ffff, 0x0080_0000,
                     MIN_ARGUMENT_BITS - 1, 0xbf80_0000, 0x7f80_0000, 0x7fc0_0000, 0xff80_0000] {
            assert!(std::panic::catch_unwind(|| index_and_exponent_shift(bits)).is_err());
        }
        assert!(std::panic::catch_unwind(|| rsqrt_from_table(1.0, &[])).is_err());
    }

    #[test]
    fn unpopulated_or_invalid_entry_is_rejected() {
        assert!(std::panic::catch_unwind(|| rsqrt_from_table(f32::from_bits(0x3f80_0002), sparse_analytic_table())).is_err());
    }
}
