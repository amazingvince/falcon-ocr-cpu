//! Standalone observed-boundary probe: rustc -O rsqrt_lookup_probe.rs -o probe.
mod rsqrt_lookup;
use std::{env, fs, io};
use rsqrt_lookup::{rsqrt_from_table, TABLE_LEN};

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args_os().collect();
    assert_eq!(args.len(), 3, "table.f32le boundary-pairs.u32le");
    let bytes = fs::read(&args[1])?;
    assert_eq!(bytes.len(), TABLE_LEN * 4);
    let table: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(table.iter().all(|x| (0.5..=1.0).contains(x)));
    let pairs = fs::read(&args[2])?;
    assert_eq!(pairs.len(), 151 * 7 * 8, "complete frozen boundary coverage required");
    let mantissas = [0, 1, 0x1f_ffff, 0x3f_ffff, 0x40_0000, 0x7f_fffe, 0x7f_ffff];
    for (index, pair) in pairs.chunks_exact(8).enumerate() {
        let argument_bits = u32::from_le_bytes(pair[..4].try_into().unwrap());
        let observed_bits = u32::from_le_bytes(pair[4..].try_into().unwrap());
        let expected_argument = (((104 + index / 7) as u32) << 23) | mantissas[index % 7];
        assert_eq!(argument_bits, expected_argument, "boundary argument order differs");
        assert_eq!(rsqrt_from_table(f32::from_bits(argument_bits), &table).to_bits(), observed_bits,
            "observed GPU output differs for input {argument_bits:08x}");
    }
    println!("{{\"status\":\"observed_boundary_values_exact\",\"table_entries\":{},\"boundary_pairs\":1057,\"gpu_work\":false}}", TABLE_LEN);
    Ok(())
}
