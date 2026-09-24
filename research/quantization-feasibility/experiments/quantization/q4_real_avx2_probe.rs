//! Std-only transport adapter for the unchanged isolated Q4 quantizer/kernels.
//! No model execution, Rayon, timing, or output-channel sampling.
mod q4_avx2;
mod q4_reference;

use q4_avx2::{Backend, linear};
use q4_reference::Q4Linear;
use std::{fs, io::Write, path::Path};

fn floats(path: &Path, expected: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != expected.checked_mul(4).ok_or("tensor length overflow")? {
        return Err("operand byte length differs".into());
    }
    let values: Vec<_> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite operand".into());
    }
    Ok(values)
}

fn write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    fs::OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)
}

fn write_floats(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_bytes(path, &bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--plan" || args[2] != "--output" {
        return Err("usage: q4_real_avx2_probe --plan CAPTURE/jobs.tsv --output NEW_DIRECTORY".into());
    }
    if !q4_avx2::avx2_available() {
        return Err("explicit AVX2/FMA unavailable".into());
    }
    let plan_path = Path::new(&args[1]);
    let root = plan_path.parent().ok_or("plan parent")?;
    let text = fs::read_to_string(plan_path)?;
    let mut lines = text.lines();
    let expected_plan = option_env!("FOCR_REAL_Q4_PLAN_SHA256").ok_or("uncaptured plan")?;
    if lines.next() != Some(&format!("FOCR_Q4_REAL_V1\t{expected_plan}")) {
        return Err("plan differs from compile-time capture".into());
    }
    let jobs: Vec<_> = lines.collect();
    if jobs.len() != 28 {
        return Err("expected all 28 projection cases".into());
    }
    let output = Path::new(&args[3]);
    fs::create_dir(output)?;
    let mut batch_cases = 0;
    let mut elements_per_backend = 0;
    for (index, job) in jobs.iter().enumerate() {
        let fields: Vec<_> = job.split('\t').collect();
        if fields.len() != 6 || fields[0].parse::<usize>()? != index {
            return Err("invalid operator inventory".into());
        }
        let n = fields[1].parse::<usize>()?;
        let k = fields[2].parse::<usize>()?;
        let source_rows = fields[3].parse::<usize>()?;
        if ![(2048, 768), (768, 1024), (4608, 768), (768, 2304)].contains(&(n, k))
            || ![1, 144].contains(&source_rows) {
            return Err("unexpected pinned fixture dimensions".into());
        }
        let weights = floats(&root.join(fields[4]), n.checked_mul(k).ok_or("shape overflow")?)?;
        let batch_specs: Vec<_> = fields[5].split(';').map(|s| s.split_once(':').ok_or("batch specification"))
            .collect::<Result<_, _>>()?;
        let expected_rows: Vec<_> = [1, 2, 4, 8].into_iter().filter(|&r| r <= source_rows).collect();
        let actual_rows = batch_specs.iter().map(|(r, _)| r.parse::<usize>()).collect::<Result<Vec<_>, _>>()?;
        if actual_rows != expected_rows {
            return Err("missing/duplicate/invented row counts".into());
        }
        let inputs: Vec<_> = batch_specs.iter().map(|(r, file)| {
            let rows = r.parse::<usize>()?;
            Ok((rows, floats(&root.join(file), rows * k)?))
        }).collect::<Result<_, Box<dyn std::error::Error>>>()?;
        for group in [64, 128] {
            let quant = Q4Linear::quantize(&weights, n, k, group)?;
            let prefix = format!("{index:03}.g{group}");
            write_bytes(&output.join(format!("{prefix}.codes.bin")), quant.packed_codes())?;
            write_floats(&output.join(format!("{prefix}.scales.f32")), quant.scales())?;
            for (rows, input) in &inputs {
                for (backend, label) in [(Backend::Scalar, "scalar"), (Backend::Avx2, "avx2")] {
                    let mut values = vec![0.0; rows * n];
                    linear(&quant, input, *rows, &mut values, backend)?;
                    write_floats(&output.join(format!("{prefix}.b{rows}.{label}.f32")), &values)?;
                }
                batch_cases += 1;
                elements_per_backend += rows * n;
            }
        }
        println!("captured operator {index:02}: N={n} K={k} source_rows={source_rows}");
    }
    let report = format!(
        "{{\"schema\":1,\"operators\":28,\"batch_cases\":{batch_cases},\"elements_per_backend\":{elements_per_backend},\"threads\":1,\"timing\":null,\"avx2_fma_available\":true,\"output_channel_sampling\":false,\"compiled_source_inventory_sha256\":\"{}\",\"compiled_plan_sha256\":\"{expected_plan}\"}}\n",
        option_env!("FOCR_REAL_Q4_SOURCE_SHA256").ok_or("uncaptured source")?);
    write_bytes(&output.join("operators.json"), report.as_bytes())?;
    Ok(())
}
