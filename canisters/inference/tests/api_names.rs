//! Public exports use only the canonical names, including feature gates.
use imajev_inference::get_candid_pointer_for_tests;

#[test]
fn only_canonical_entrypoints_are_exported() {
    let service = get_candid_pointer_for_tests();
    let methods = service.rsplit("service :").next().unwrap();
    let pairs = [
        ("prepare", "prepareModelUpload", true),
        ("upload", "uploadModelBytes", false),
        ("upload_chunk", "uploadModelChunk", true),
        ("hash_pack", "verifyModelUpload", true),
        ("seal", "finalizeModelUpload", false),
        ("pack_status", "getModelStatus", true),
        ("status", "getModelUploadProgress", false),
        ("step", "runInferenceStep", true),
        ("profile_step", "profileInferenceStep", cfg!(feature = "paid-update-diagnostics")),
        ("decision", "getInferenceDecision", false),
        ("decision_fast", "getCandidateDecision", false),
        ("terminal_step_decision", "runFinalInferenceStep", true),
        ("mlp_delta_front", "runDeltaInferenceStep", cfg!(feature = "experimental-mlp-delta-query")),
        ("attention_mlp_front", "runAttentionInferenceStep", false),
        ("warm_weights", "prepareWeightCache", true),
        ("clear_weight_cache", "clearWeightCache", true),
        ("weight_cache_status", "getWeightCacheStatus", true),
        ("update_prefix", "installInferencePrefix", cfg!(feature = "experimental-update-inference")),
        ("update_infer_start", "startOwnerInference", cfg!(feature = "paid-update-diagnostics")),
        ("update_infer_continue", "continueOwnerInference", cfg!(feature = "paid-update-diagnostics")),
        ("quote", "getInferenceQuote", cfg!(feature = "paid-update-inference")),
        ("infer", "runPaidInference", cfg!(feature = "paid-update-inference")),
        ("configure_paid", "configurePaidInference", cfg!(feature = "paid-update-inference")),
        ("paid_config", "getPaidInferenceConfig", cfg!(feature = "paid-update-inference")),
        ("inference_status", "getInferenceReceipt", cfg!(feature = "paid-update-inference")),
        ("inference_step", "runPaidInferenceWorker", cfg!(feature = "paid-update-inference")),
        ("retry_inference_refund", "retryInferenceRefund", cfg!(feature = "paid-update-inference")),
        ("paid_debug", "getPaidInferenceDebug", cfg!(feature = "paid-update-diagnostics")),
        ("prepare_fixed_prefix_state", "prepareFixedPrefixCache", false),
        ("paid_fault", "setPaidInferenceFault", cfg!(feature = "paid-update-diagnostics")),
        ("paid_probe_step", "probePaidInferenceWorker", cfg!(feature = "paid-update-diagnostics")),
        ("paid_upgrade_probe", "preparePaidInferenceUpgradeProbe", cfg!(feature = "paid-update-diagnostics")),
        ("paid_reference", "setPaidInferenceReference", cfg!(feature = "paid-update-diagnostics")),
    ];
    let mut expected_count = 0;
    for (old, new, enabled) in pairs {
        assert!(!methods.contains(&format!("\n  {old} : ")), "legacy export: {old}");
        assert_eq!(methods.contains(&format!("\n  {new} : ")), enabled, "feature gate: {new}");
        if enabled {
            expected_count += 1;
            let declaration = methods.split_once(&format!("\n  {new} : ")).unwrap().1.split(';').next().unwrap();
            let query = ["getModelStatus", "getWeightCacheStatus", "runInferenceStep", "runDeltaInferenceStep",
                         "runAttentionInferenceStep", "runFinalInferenceStep", "getInferenceQuote", "getInferenceReceipt",
                         "getPaidInferenceConfig", "profileInferenceStep", "getPaidInferenceDebug"].contains(&new);
            assert_eq!(declaration.trim_end().ends_with("query"), query, "method mode: {new}");
        }
    }
    let exported = methods.lines().filter(|line| line.starts_with("  ") && line.contains(" : ")).count();
    assert_eq!(exported, expected_count, "unexpected public methods");
}

#[test]
fn paid_inference_contract_does_not_expose_fee_versions() {
    if !cfg!(feature="paid-update-inference") {return;}
    let service=get_candid_pointer_for_tests();
    assert!(!service.contains("quote_version"));
    assert!(!service.contains("enabled :"));
    assert!(!service.contains("Paused"));
    assert!(!service.contains("QuoteChanged"));
    let method=service.split_once("runPaidInference : ").unwrap().1.split(';').next().unwrap();
    assert!(method.starts_with("(InferRequest, text)"),"{method}");
    for name in ["Quote", "Config"] {
        let record=service.split_once(&format!("type {name} = record {{")).unwrap().1.split('}').next().unwrap();
        assert!(!record.contains("version :"),"fee version in {name}");
    }
}
