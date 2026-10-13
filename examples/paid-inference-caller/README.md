# Calling paid Imajev inference

`lib.rs` is an owner-controlled relay for tests and integration. Only its owner can call `forward`. It forwards Candid arguments and cycles to a selected canister and returns the raw response, unaccepted cycles and execution balances before and after the call. It does not create prepaid credit accounts.

The service types are in `canisters/inference/src/paid_types.rs`; the default-feature Candid contract is `canisters/inference/paid-inference.did`. `InferRequest` contains only `token_ids` and `options`. Prepare the full prompt with the pinned tokenizer; the service uses its installed model and fixed prefix. Use `getInferenceQuote` to obtain the applicable prefix/suffix lengths, fee and execution limits. Browser query limits and archived token limits do not specify the limits of an arbitrary paid deployment.

```rust
use ic_cdk::call::Call;

let quote: Result<Quote, InferError> = Call::unbounded_wait(service, "getInferenceQuote")
    .with_arg(request.clone())
    .await?
    .candid()?;
let quote = quote.map_err(|e| format!("{e:?}"))?;
let result: Result<InferenceResult, InferError> = Call::unbounded_wait(service, "runPaidInference")
    .with_args(&(request, request_id))
    .with_cycles(quote.fee)
    .await?
    .candid()?;
```

Quotes can be reused while input shape and pricing stay the same. Admission recomputes the current fee and rejects insufficient attached cycles before charging. Only the required amount is accepted; the IC returns excess cycles. Ordinary browser update calls cannot attach cycles directly, so use a calling canister, wallet or relay.

The service runs sequential self-call workers and returns the final result to the original call; callers do not manually repeat start/continue. Invalid input, Busy and insufficient cycles do not incur a service fee, although caller-side execution and messaging costs remain. Worker failure after charging triggers an explicit full refund. If a refund is Pending, the same caller can invoke `retryInferenceRefund(request_id)`; do not automatically resend an InFlight refund.

Choose a unique request ID per caller, at most 64 bytes. Repeating the same input and ID returns the retained receipt without charging again. Reusing an ID for different input gives IdConflict. Receipts are bounded to 128 entries and 24 hours, with unresolved refunds retained beyond expiry. Full receipt storage rejects new work before charging. Completed/failed receipts, refunds and pricing survive upgrades; active jobs and InFlight refunds block upgrades. Heap weight/prefix caches need preparation after upgrades, while completed receipts can be retrieved before cache readiness.

[poc-config.json](poc-config.json) records test pricing: 10B base cycles plus 3B per suffix token. `scripts/configure_paid_poc.py` applies configuration to an explicitly selected local canister and reads it back. Production pricing requires measurement on the intended subnet. `reserve_cycles` is extra liquid-balance headroom beyond freezing costs.

Build Candid input/output helpers with:

```sh
CARGO_INCREMENTAL=0 cargo build --locked -p imajev-client --example paid_update_args
```

Fault/probe methods require `paid-update-diagnostics` and are omitted from normal releases. A standard Cargo build differs from archived optimized Wasm builds. Stop/upgrade only after inference and refunds finish. See the [API guide](../../docs/CANISTER_API_NAMES.md) and [archive guide](../../docs/ARCHIVE.md).
