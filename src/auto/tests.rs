//! Tests of the automatic configuration: modes, weights resolution, the plan
//! and `doctor`.
use super::*;

fn fabricate_packed(path: &Path, profile: &str) {
    let data = [0u8; 4];
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data).unwrap();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("format".to_owned(), crate::model::PACKED_FORMAT.to_owned());
    metadata.insert("profile".to_owned(), profile.to_owned());
    safetensors::serialize_to_file(vec![("t", view)], Some(metadata), path).unwrap();
}

#[test]
fn mode_labels_profiles_and_packed_names_round_trip() {
    for mode in [Mode::Exact, Mode::NearExact, Mode::Fast] {
        assert_eq!(Mode::of_profile(mode.profile(true)), Some(mode));
        assert_eq!(Mode::of_profile(mode.profile(false)), Some(mode));
        assert_eq!(serde_json::to_value(mode).unwrap(), serde_json::json!(mode.label()));
    }
    assert_eq!(Mode::Exact.profile(true), Profile::SPLIT_F32);
    assert_eq!(Mode::Exact.profile(false), Profile::REFERENCE);
    assert_eq!(Mode::of_profile(Profile::W8_BODY), None);
    assert_eq!(Mode::Exact.packed_file_name(), None);
    assert!(Mode::Fast.packed_file_name().unwrap().contains("fast"));
}

#[test]
fn the_published_draft_head_drafts_unless_a_drafter_was_chosen() {
    let dir = tempfile::tempdir().unwrap();
    let base = RunnerConfig::default();
    assert!(base.speculation.is_some() && base.draft_head.is_none());
    // No head file: nothing changes.
    let config = with_default_draft_head(base.clone(), dir.path(), false);
    assert_eq!((config.drafter, config.draft_head), (base.drafter, None));
    std::fs::write(dir.path().join(DRAFT_HEAD_FILE), b"stub").unwrap();
    let config = with_default_draft_head(base.clone(), dir.path(), false);
    assert_eq!(config.drafter, Drafter::Head);
    assert_eq!(config.draft_head, Some(dir.path().join(DRAFT_HEAD_FILE)));
    // An explicit --drafter/--draft-head, or no speculation, keeps the choice.
    assert_eq!(with_default_draft_head(base.clone(), dir.path(), true).draft_head, None);
    let off = RunnerConfig {
        speculation: None,
        ..base
    };
    assert_eq!(with_default_draft_head(off, dir.path(), false).draft_head, None);
    // The folder of --model-file, else --model.
    let file = dir.path().join("falcon-ocr-v1.5-fast.safetensors");
    let request = ModelRequest {
        model_dir: Path::new("elsewhere"),
        model_file: Some(&file),
        ..ModelRequest::default()
    };
    assert_eq!(model_files_dir(&request), dir.path());
    let request = ModelRequest {
        model_dir: dir.path(),
        ..ModelRequest::default()
    };
    assert_eq!(model_files_dir(&request), dir.path());
}

#[test]
fn lookup_prefers_packed_files_then_the_checkpoint_and_names_remedies() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    // Nothing there: every mode fails with a remedy.
    let request = ModelRequest {
        model_dir: root,
        ..ModelRequest::default()
    };
    let error = resolve_weights(&request).unwrap_err().to_string();
    assert!(error.contains("no near-exact model"), "{error}");
    assert!(
        error.contains("huggingface-cli") && error.contains("fetch_reference"),
        "{error}"
    );
    // A checkpoint alone: near-exact quantizes at load, fast needs the overlay.
    std::fs::write(root.join("model.safetensors"), b"stub").unwrap();
    let plan = resolve_weights(&request).unwrap();
    assert_eq!(
        (plan.mode, plan.profile),
        (Some(Mode::NearExact), Profile::W16_BODY_KV_Q16)
    );
    assert!(matches!(
        plan.source,
        WeightsSource::Checkpoint {
            overlay: None,
            rtn: false,
            ..
        }
    ));
    let fast = ModelRequest {
        model_dir: root,
        mode: Some(Mode::Fast),
        ..ModelRequest::default()
    };
    let error = resolve_weights(&fast).unwrap_err().to_string();
    assert!(
        error.contains("--allow-rtn") && error.contains("make_gptq_overlay"),
        "{error}"
    );
    let rtn = ModelRequest {
        allow_rtn: true,
        ..fast.clone()
    };
    assert!(matches!(
        resolve_weights(&rtn).unwrap().source,
        WeightsSource::Checkpoint {
            rtn: true,
            overlay: None,
            ..
        }
    ));
    std::fs::write(root.join(DEFAULT_OVERLAY), b"stub").unwrap();
    assert!(matches!(
        resolve_weights(&fast).unwrap().source,
        WeightsSource::Checkpoint {
            rtn: false,
            overlay: Some(_),
            ..
        }
    ));
    assert!(
        resolve_weights(&ModelRequest {
            w8_artifact: Some(Path::new("x")),
            ..request.clone()
        })
        .is_err()
    );
    // Exact: the split cache for single pages, the reference cache otherwise.
    let exact = ModelRequest {
        model_dir: root,
        mode: Some(Mode::Exact),
        split_exact: true,
        ..ModelRequest::default()
    };
    assert_eq!(resolve_weights(&exact).unwrap().profile, Profile::SPLIT_F32);
    let exact = ModelRequest {
        split_exact: false,
        ..exact
    };
    assert_eq!(resolve_weights(&exact).unwrap().profile, Profile::REFERENCE);
    // A packed file next to the checkpoint wins for its mode, unless the
    // caller wants the checkpoint (pack).
    let packed = root.join(Mode::NearExact.packed_file_name().unwrap());
    fabricate_packed(&packed, "w16-body-kv-q16");
    let plan = resolve_weights(&request).unwrap();
    assert_eq!(plan.source, WeightsSource::Packed { path: packed.clone() });
    assert_eq!(
        (plan.mode, plan.profile),
        (Some(Mode::NearExact), Profile::W16_BODY_KV_Q16)
    );
    assert!(matches!(
        resolve_weights(&ModelRequest {
            from_checkpoint: true,
            ..request.clone()
        })
        .unwrap()
        .source,
        WeightsSource::Checkpoint { .. }
    ));
    // A mislabelled packed file is refused.
    fabricate_packed(&packed, "w8-body-kv-q8");
    let error = resolve_weights(&request).unwrap_err().to_string();
    assert!(error.contains("not a near-exact"), "{error}");
    // An explicit file decides the mode; a disagreeing --mode is an error.
    let file = root.join("custom.safetensors");
    fabricate_packed(&file, "w8-body-kv-q8");
    let explicit = ModelRequest {
        model_dir: root,
        model_file: Some(&file),
        ..ModelRequest::default()
    };
    let plan = resolve_weights(&explicit).unwrap();
    assert_eq!(plan.mode, Some(Mode::Fast));
    let conflict = ModelRequest {
        mode: Some(Mode::NearExact),
        ..explicit.clone()
    };
    let error = resolve_weights(&conflict).unwrap_err().to_string();
    assert!(error.contains("--mode near-exact"), "{error}");
    fabricate_packed(&file, "w8-body");
    let error = resolve_weights(&explicit).unwrap_err().to_string();
    assert!(error.contains("research profile"), "{error}");
    let research = resolve_weights(&ModelRequest {
        allow_research: true,
        ..explicit.clone()
    })
    .unwrap();
    assert_eq!((research.mode, research.profile), (None, Profile::W8_BODY));
    // A named research profile reads the checkpoint (8-bit: overlay or RTN).
    let named = ModelRequest {
        model_dir: root,
        profile: Some(Profile::KV_Q8),
        ..ModelRequest::default()
    };
    let named_plan = resolve_weights(&named).unwrap();
    assert_eq!((named_plan.mode, named_plan.profile), (None, Profile::KV_Q8));
    assert!(matches!(
        named_plan.source,
        WeightsSource::Checkpoint {
            rtn: false,
            overlay: None,
            ..
        }
    ));
    let named = ModelRequest {
        profile: Some(Profile::W8_BODY_KV_Q8),
        ..named
    };
    let named_plan = resolve_weights(&named).unwrap();
    assert_eq!(named_plan.mode, Some(Mode::Fast));
    assert!(matches!(
        named_plan.source,
        WeightsSource::Checkpoint { overlay: Some(_), .. }
    ));
    assert!(
        resolve_weights(&ModelRequest {
            mode: Some(Mode::Exact),
            ..named
        })
        .is_err()
    );
    // The tokenizer comes from the file's folder only when it holds one.
    assert_eq!(
        plan.source.tokenizer_dir(Path::new("elsewhere")),
        PathBuf::from("elsewhere")
    );
    std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
    assert_eq!(plan.source.tokenizer_dir(Path::new("elsewhere")), root.to_path_buf());
}

/// `--w8-artifact` is used even when a packed fast file, which would
/// otherwise win, sits in the model directory.
#[test]
fn an_explicit_overlay_reads_the_checkpoint_over_a_packed_file() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fabricate_packed(&root.join(Mode::Fast.packed_file_name().unwrap()), "w8-body-kv-q8");
    let fast = ModelRequest {
        model_dir: root,
        mode: Some(Mode::Fast),
        ..ModelRequest::default()
    };
    assert!(matches!(
        resolve_weights(&fast).unwrap().source,
        WeightsSource::Packed { .. }
    ));
    let overlay = root.join("w8-gptq-other.safetensors");
    let with_overlay = ModelRequest {
        w8_artifact: Some(&overlay),
        ..fast
    };
    // Without a checkpoint the overlay has nothing to apply to.
    let error = resolve_weights(&with_overlay).unwrap_err().to_string();
    assert!(
        error.contains("applies to the FP32 checkpoint") && error.contains("drop --w8-artifact"),
        "{error}"
    );
    std::fs::write(root.join("model.safetensors"), b"stub").unwrap();
    let plan = resolve_weights(&with_overlay).unwrap();
    assert_eq!(
        plan.source,
        WeightsSource::Checkpoint {
            dir: root.to_path_buf(),
            overlay: Some(overlay.clone()),
            rtn: false
        }
    );
    assert_eq!((plan.mode, plan.profile), (Some(Mode::Fast), Profile::W8_BODY_KV_Q8));
}

#[test]
fn byte_budget_matches_the_documented_figures() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
    let fast = byte_budget(&c, Profile::W8_BODY_KV_Q8, HeadMode::Screened);
    // 168.7M body parameters as INT8 plus FP32 scales per 64; the INT8
    // screen of the 50.3M-parameter head plus its scales.
    assert_eq!(fast.weights_per_token, 179_232_768);
    assert_eq!(fast.head_per_token, 53_477_376);
    // 22 layers x 8 groups x 170 bytes: 196 MB at the 6,544-token journal page.
    assert_eq!(fast.kv_per_position, 29_920);
    let near = byte_budget(&c, Profile::W16_BODY_KV_Q16, HeadMode::Screened);
    assert_eq!(near.weights_per_token, 2 * 168_689_664 + 10_543_104);
    assert_eq!(near.kv_per_position, 22 * 8 * 330);
    let exact = byte_budget(&c, Profile::SPLIT_F32, HeadMode::Full);
    assert_eq!(exact.weights_per_token, 4 * 168_689_664);
    assert_eq!(exact.head_per_token, 4 * 65536 * 768);
    assert_eq!(exact.kv_per_position, 22 * 8 * 640);
    assert_eq!(
        byte_budget(&c, Profile::REFERENCE, HeadMode::Full).kv_per_position,
        22 * 1536 * 4
    );
    // Rotation adds no bytes; 4-bit codes take half a byte per element.
    let rotated = byte_budget(&c, Profile::W8_BODY_KV_Q8R, HeadMode::Screened);
    assert_eq!(rotated, fast);
    assert_eq!(
        byte_budget(&c, Profile::W8_BODY_KV_Q4R, HeadMode::Screened).kv_per_position,
        22 * 8 * 90
    );
    // Near-exact with the rotated 8-bit cache at the 6,544-position journal
    // page: 23% fewer bytes per decode step.
    let cheaper = byte_budget(&c, Profile::W16_BODY_KV_Q8R, HeadMode::Screened);
    let step = |b: &ByteBudget| b.weights_per_token + b.head_per_token + 6544 * b.kv_per_position;
    let saved = 1.0 - step(&cheaper) as f64 / step(&near) as f64;
    assert!((0.23..0.24).contains(&saved), "{saved}");
}

#[test]
fn kv_cache_replaces_the_kv_half_of_the_resolved_profile() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("model.safetensors"), b"stub").unwrap();
    let request = |mode, kv_cache| ModelRequest {
        model_dir: root,
        mode,
        kv_cache,
        allow_rtn: true,
        split_exact: true,
        ..ModelRequest::default()
    };
    // A mode's own cache keeps the mode; any other makes a research profile.
    let plan = resolve_weights(&request(Some(Mode::Fast), Some(Kv::Q8))).unwrap();
    assert_eq!((plan.mode, plan.profile), (Some(Mode::Fast), Profile::W8_BODY_KV_Q8));
    let plan = resolve_weights(&request(Some(Mode::Fast), Some(Kv::Q16))).unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::W8_BODY_KV_Q16));
    assert!(matches!(plan.source, WeightsSource::Checkpoint { rtn: true, .. }));
    let plan = resolve_weights(&request(None, Some(Kv::Q8))).unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::W16_BODY_KV_Q8));
    let plan = resolve_weights(&request(Some(Mode::Exact), Some(Kv::Compact))).unwrap();
    assert_eq!((plan.mode, plan.profile), (Some(Mode::Exact), Profile::REFERENCE));
    let plan = resolve_weights(&request(Some(Mode::Exact), Some(Kv::Q8))).unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::KV_Q8));
    let plan = resolve_weights(&request(Some(Mode::Fast), Some(Kv::Q8Rot))).unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::W8_BODY_KV_Q8R));
    let plan = resolve_weights(&request(None, Some(Kv::Q4Rot))).unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::W16_BODY_KV_Q4R));
    // The plan reports the profile that runs.
    std::fs::write(
        root.join("config.json"),
        include_str!("../../tests/fixtures/model-config.json"),
    )
    .unwrap();
    let plan = resolve_weights(&request(Some(Mode::Fast), Some(Kv::Q16))).unwrap();
    let facts = ModelFacts::from_plan(&plan).unwrap();
    let resolved = Resolved::new(HostInfo::detect(), &RunnerConfig::default(), &facts, root);
    assert_eq!((resolved.mode, resolved.profile), (None, Profile::W8_BODY_KV_Q16));
    assert_eq!(resolved.kv_cache, "q16");
    assert_eq!(resolved.bytes.kv_per_position, 22 * 8 * 330);
    assert!(resolved.to_string().contains("mode research (w8-body-kv-q16)"));
    // A kernel-ready file keeps its weights and takes the requested cache.
    let packed = root.join(Mode::Fast.packed_file_name().unwrap());
    fabricate_packed(&packed, "w8-body-kv-q8");
    let plan = resolve_weights(&request(Some(Mode::Fast), Some(Kv::Q16))).unwrap();
    assert_eq!(plan.source, WeightsSource::Packed { path: packed.clone() });
    assert_eq!((plan.mode, plan.profile), (None, Profile::W8_BODY_KV_Q16));
    let explicit = ModelRequest {
        model_file: Some(&packed),
        ..request(None, Some(Kv::Q16))
    };
    let plan = resolve_weights(&explicit).unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::W8_BODY_KV_Q16));
    // A research profile with a model file (the eval binary's doctor)
    // must name the file's weights; its cache comes from `kv_cache`.
    let research = |profile| ModelRequest {
        profile: Some(profile),
        allow_research: true,
        ..explicit.clone()
    };
    let plan = resolve_weights(&ModelRequest {
        kv_cache: Some(Kv::F32Split),
        ..research(Profile::W8_BODY_SPLIT_F32)
    })
    .unwrap();
    assert_eq!(plan.profile, Profile::W8_BODY_SPLIT_F32);
    let error = resolve_weights(&research(Profile::W16_BODY_KV_Q8))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("w8-body-kv-q8 weights, not those of w16-body-kv-q8"),
        "{error}"
    );
}

#[test]
fn doctor_reports_files_and_a_plan_without_reading_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let config = RunnerConfig::default();
    let request = ModelRequest {
        model_dir: root,
        ..ModelRequest::default()
    };
    let report = doctor(&request, &config, false, false).unwrap();
    assert!(report.plan.is_none() && report.error.as_deref().unwrap().contains("no near-exact model"));
    assert!(!report.files.checkpoint.present && report.files.model_file.is_none());
    assert!(report.to_string().contains("plan: none"));
    // A checkpoint directory: the plan comes from config.json.
    std::fs::write(root.join("model.safetensors"), b"stub").unwrap();
    std::fs::write(
        root.join("config.json"),
        include_str!("../../tests/fixtures/model-config.json"),
    )
    .unwrap();
    let report = doctor(&request, &config, false, false).unwrap();
    let plan = report.plan.as_ref().unwrap();
    assert_eq!(plan.mode, Some(Mode::NearExact));
    assert_eq!(plan.kv_cache, "q16");
    assert_eq!(plan.bytes.weights_per_token, 2 * 168_689_664 + 10_543_104);
    assert!(report.files.checkpoint.present && report.error.is_none());
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["plan"]["profile"], "w16-body-kv-q16");
    assert!(report.to_string().starts_with("host: "));
    // A replaced cache shows in the plan.
    let replaced = ModelRequest {
        kv_cache: Some(Kv::Q8),
        ..request.clone()
    };
    let plan = doctor(&replaced, &config, false, false).unwrap().plan.unwrap();
    assert_eq!((plan.mode, plan.profile), (None, Profile::W16_BODY_KV_Q8));
    assert_eq!(plan.kv_cache, "q8");
    assert_eq!(plan.bytes.kv_per_position, 22 * 8 * 170);
}

#[test]
fn host_info_is_consistent() {
    let host = HostInfo::detect();
    assert!(host.physical_cores >= 1 && host.physical_cores <= host.logical_cpus);
    assert_eq!(host.smt, host.logical_cpus > host.physical_cores);
    if let Some(p) = host.performance_cores {
        assert!(p >= 1 && p < host.physical_cores);
    }
    #[cfg(target_arch = "x86_64")]
    assert_eq!(host.features.avx2, std::is_x86_feature_detected!("avx2"));
    assert!(serde_json::to_string(host).unwrap().contains("\"logical_cpus\""));
}
