//! `falcon-ocr doctor`: the host, the files found in the model directory, the
//! plan for a request, and optionally a load and a bandwidth probe.
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    DEFAULT_OVERLAY, DRAFT_HEAD_FILE, HostInfo, Mode, ModelFacts, ModelRequest, Resolved, load_model, model_files_dir,
    resolve_weights,
};
use crate::config::{HeadMode, RunnerConfig};

/// One file `doctor` looked for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStatus {
    pub path: PathBuf,
    pub present: bool,
    pub bytes: Option<u64>,
}

impl FileStatus {
    fn of(path: PathBuf) -> Self {
        let bytes = std::fs::metadata(&path).ok().filter(|m| m.is_file()).map(|m| m.len());
        Self {
            present: bytes.is_some(),
            bytes,
            path,
        }
    }
}

/// The files `doctor` looked for in the model directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Files {
    pub model_file: Option<FileStatus>,
    pub packed_near_exact: FileStatus,
    pub packed_fast: FileStatus,
    pub checkpoint: FileStatus,
    pub overlay: FileStatus,
    pub tokenizer: FileStatus,
    #[serde(default = "missing_draft_head")]
    pub draft_head: FileStatus,
}

fn missing_draft_head() -> FileStatus {
    FileStatus::of(PathBuf::from(DRAFT_HEAD_FILE))
}

/// `doctor --load`: the model was loaded and its screened head built.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    pub load_ms: f64,
    pub screened_head_bytes: usize,
}

/// `doctor --probe`: memory read bandwidth and the decode floor it implies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub bytes: usize,
    pub threads: usize,
    /// GB/s of each pass.
    pub gb_per_s: Vec<f64>,
    pub median_gb_per_s: f64,
    /// Milliseconds the plan's per-token weight and head bytes take at the
    /// median bandwidth: no decode step can be faster.
    pub decode_floor_ms: Option<f64>,
}

/// What `doctor` reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Doctor {
    pub host: HostInfo,
    pub files: Files,
    /// The plan for the request, when its weights were found.
    pub plan: Option<Resolved>,
    /// Why there is no plan.
    pub error: Option<String>,
    pub load: Option<LoadReport>,
    pub probe: Option<ProbeReport>,
}

/// What `auto` would do for `request` and `config` on this host, without
/// reading tensors unless `load`; `probe` measures memory bandwidth (512 MB,
/// five passes on the physical cores).
pub fn doctor(request: &ModelRequest<'_>, config: &RunnerConfig, load: bool, probe: bool) -> Result<Doctor> {
    config.validate()?;
    let host = HostInfo::detect().clone();
    let dir = request.model_dir;
    let files = Files {
        model_file: request.model_file.map(|p| FileStatus::of(p.to_path_buf())),
        packed_near_exact: FileStatus::of(dir.join(Mode::NearExact.packed_file_name().unwrap_or_default())),
        packed_fast: FileStatus::of(dir.join(Mode::Fast.packed_file_name().unwrap_or_default())),
        checkpoint: FileStatus::of(dir.join("model.safetensors")),
        overlay: FileStatus::of(
            request
                .w8_artifact
                .map_or_else(|| dir.join(DEFAULT_OVERLAY), Path::to_path_buf),
        ),
        tokenizer: FileStatus::of(dir.join("tokenizer.json")),
        draft_head: FileStatus::of(model_files_dir(request).join(DRAFT_HEAD_FILE)),
    };
    let (mut plan, mut error, mut load_report) = (None, None, None);
    match resolve_weights(request) {
        Err(e) => error = Some(format!("{e:#}")),
        Ok(weights) => {
            let tokenizer_dir = weights.source.tokenizer_dir(dir);
            plan = Some(Resolved::new(
                &host,
                config,
                &ModelFacts::from_plan(&weights)?,
                &tokenizer_dir,
            ));
            if load {
                let started = Instant::now();
                let model = load_model(&weights, false)?;
                if config.head == HeadMode::Screened {
                    model.prepare_screened_head()?;
                }
                load_report = Some(LoadReport {
                    load_ms: started.elapsed().as_secs_f64() * 1000.0,
                    screened_head_bytes: model.screened_head_bytes(),
                });
                plan = Some(Resolved::new(&host, config, &model.facts(), &tokenizer_dir));
            }
        }
    }
    let probe = probe.then(|| {
        let bytes = 512 << 20;
        let threads = host.physical_cores;
        let gb_per_s = crate::cpu::read_bandwidth_gb_s(bytes, threads, 5);
        let median_gb_per_s = crate::tune::median(&gb_per_s);
        ProbeReport {
            bytes,
            threads,
            decode_floor_ms: plan
                .as_ref()
                .map(|p| (p.bytes.weights_per_token + p.bytes.head_per_token) as f64 / (median_gb_per_s * 1e9) * 1e3),
            gb_per_s,
            median_gb_per_s,
        }
    });
    Ok(Doctor {
        host,
        files,
        plan,
        error,
        load: load_report,
        probe,
    })
}

impl std::fmt::Display for Doctor {
    /// The text form (`doctor --text`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let h = &self.host;
        writeln!(
            f,
            "host: {} {}, {} logical / {} physical cores{}{}; {}",
            h.os,
            h.arch,
            h.logical_cpus,
            h.physical_cores,
            if h.smt { " (SMT)" } else { "" },
            h.performance_cores
                .map_or(String::new(), |p| format!(", {p} performance cores")),
            h.features.names().join(" ")
        )?;
        let files = [
            ("model file", self.files.model_file.as_ref()),
            ("packed near-exact", Some(&self.files.packed_near_exact)),
            ("packed fast", Some(&self.files.packed_fast)),
            ("checkpoint", Some(&self.files.checkpoint)),
            ("overlay", Some(&self.files.overlay)),
            ("tokenizer", Some(&self.files.tokenizer)),
            ("draft head", Some(&self.files.draft_head)),
        ];
        for (name, file) in files.into_iter().filter_map(|(n, f)| f.map(|f| (n, f))) {
            let size = match file.bytes {
                Some(b) => format!("({:.0} MB)", b as f64 / 1e6),
                None => "(missing)".to_owned(),
            };
            writeln!(f, "file: {name:18} {} {size}", file.path.display())?;
        }
        match (&self.plan, &self.error) {
            (Some(plan), _) => writeln!(f, "plan: {plan}")?,
            (None, Some(error)) => writeln!(f, "plan: none ({error})")?,
            (None, None) => {}
        }
        if let Some(load) = &self.load {
            writeln!(
                f,
                "load: {:.0} ms, screened head {:.0} MB",
                load.load_ms,
                load.screened_head_bytes as f64 / 1e6
            )?;
        }
        if let Some(probe) = &self.probe {
            writeln!(
                f,
                "probe: {} MB x{} on {} threads: {:.1} GB/s median{}",
                probe.bytes >> 20,
                probe.gb_per_s.len(),
                probe.threads,
                probe.median_gb_per_s,
                probe
                    .decode_floor_ms
                    .map_or(String::new(), |ms| format!("; decode floor {ms:.2} ms/token"))
            )?;
        }
        Ok(())
    }
}
