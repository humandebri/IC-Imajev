# Canister API

Public methods use verb-plus-object camelCase. Browser step queries and paid updates are separate interfaces. Optimized releases expose 20 production methods; the standard Cargo build exposes 18 because two specialized Attention/prefix handlers require optimized sources. Feature-disabled builds have a smaller service. These counts describe the documented source/build configurations, not a live deployment check.

| Method | Access | Purpose |
| --- | --- | --- |
| `runInferenceStep` | public query | Advance one frame |
| `runDeltaInferenceStep` | public query | Finish MLP, run the next Delta layer and begin the next MLP |
| `runAttentionInferenceStep` | public query | Finish MLP, run the next Attention layer and begin the next MLP |
| `runFinalInferenceStep` | public query | Finish the final stage and return the decision |
| `getModelStatus` | public query | Model/pack identity and upload progress |
| `getWeightCacheStatus` | public query | Immutable weight-cache readiness |
| `getInferenceQuote` | public query | Paid-inference price and limits |
| `runPaidInference` | update | Execute inference with caller-supplied cycles |
| `getInferenceReceipt` | caller-scoped query | Retrieve the caller's receipt |
| `retryInferenceRefund` | caller-scoped update | Retry a Pending refund |
| `prepareModelUpload` | owner update | Register a manifest and prepare upload |
| `uploadModelChunk` | owner update | Upload fixed chunks with duplicate checks |
| `verifyModelUpload` | owner update | Hash a bounded number of bytes and check completion |
| `prepareWeightCache` | owner update | Prepare immutable weight caches |
| `clearWeightCache` | owner update | Clear the weight cache |
| `prepareFixedPrefixCache` | owner update | Prepare the optimized release's fixed prefix |
| `installInferencePrefix` | owner update | Install per-layer prefixes |
| `getPaidInferenceConfig` | public query | Read pricing configuration |
| `configurePaidInference` | owner update | Configure pricing and operational headroom |
| `runPaidInferenceWorker` | self-only update | Advance a paid job in a separate message |

Public queries retain their existing owner checks when executed through replicated ingress. Inference blobs use the model-specific frame/carry protocol; they are not UTF-8 question strings. Lifecycle hooks and internal Rust names are separate from Candid names.

## Service definitions and validation

`canisters/inference/paid-inference.did` describes the default-feature service. The generated `inference.did` describes a feature-disabled service. `public-query.did` contains the six browser queries; `optimized-inference.did` describes the optimized production service. Never publish a subset as though it were the entire service.

`paid-update-diagnostics` adds eight owner diagnostic methods: `profileInferenceStep`, `startOwnerInference`, `continueOwnerInference`, `getPaidInferenceDebug`, `setPaidInferenceFault`, `setPaidInferenceReference`, `probePaidInferenceWorker` and `preparePaidInferenceUpgradeProbe`. Normal releases omit their Wasm exports. Detailed profiling also requires `instruction-profile`.

`scripts/canister_api_names.py` applies renaming, removal and diagnostic gating to current and frozen sources. `canisters/inference/tests/api_names.rs`, `scripts/test_canister_api_names.py` and `scripts/check_canister_api_exports.py` validate names, counts, query/update classification and gating. Optimized builds derive `service.did` from their actual Wasm and check it against exports.

Removed methods include `getModelUploadProgress`, `uploadModelBytes`, `finalizeModelUpload`, `getInferenceDecision` and `getCandidateDecision`. Use model status, chunk upload/hash verification, and final inference results instead. Upload resumes through the chunk API; reset must be explicit. An upgrade resets incomplete upload counters and heap chunk bookkeeping; unfinished chunks must be resent and verified. Verified model data remains stored.

## Paid arguments and upgrades

`runPaidInference` accepts an `InferRequest` and a `request_id`. The request contains `token_ids` and `options`, without model, version or pricing-version arguments. Admission computes the current fee and rejects insufficient cycles before charging. The accepted amount is retained for accounting and refunds. Older stored upgrade JSON may still contain historical pricing-version fields. See the [caller example](../examples/paid-inference-caller/README.md).

Coordinate a canister upgrade with heap-cache preparation, frontend module/prefix pins and focused inference checks. Builds and documentation edits do not deploy a canister. Historical frozen-source measurements are in the [archive](ARCHIVE.md).
