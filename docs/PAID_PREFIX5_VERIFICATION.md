# Current paid inference verification

The supported execution path uses camelCase API exports, the five-token prefix
`[248045, 846, 198, 1349, 25]`, and the current Cargo Candid helpers.
There is no legacy transport mode or fee-version argument. Saved prefix27
experiments remain historical evidence; their entrypoints stop before reading
assets or making network calls.

## Build and prepare

```sh
cargo build -p imajev-client --examples
.venv/bin/python -B scripts/build_paid_token_chunks.py \
  --source-root . --directory artifacts/paid-prefix5/build --diagnostics
```

This builder retains the validated numerical kernels and generates a current
`service.did` from the resulting WASM. It binds the frozen fixed-cache length to
the paid prefix and takes the Delta stream implementation from current source.
The stream bounds the entire sequence, including the prefix, to 1024 tokens when token chunks are enabled (512 otherwise).
The unchunked suffix bound stays 89 (94 total); longer inputs use token tiles.
`TextPreparer` prepares real text up to 1024 total tokens using the same
`paid_prefix.MAX_TOKENS` bound as the paid fixtures. The browser independently
enforces its 96-token release limit; preparing text does not change a canister's
admission limit.
Attention keeps eight heads per group when its work and buffer bounds allow it,
and uses four heads for longer histories. Per-tile row limits stay unchanged.
The local call helper polls up to 30 minutes so a long inference remains one
update submission; it does not resubmit on a polling timeout.

`paid_raw_call` reads the current gateway URL from `icp network status --json`
when no URL is supplied. Override it with `--url` or `IMAJEV_LOCAL_URL`.
Use `--identity-pem` or `IMAJEV_LOCAL_IDENTITY_PEM` for a different PEM file;
the default remains `artifacts/imajev-local.pem`. Only loopback HTTP(S) URLs
are accepted because this helper fetches the local root key.
Python `PaidTransport` also accepts `url=` and `identity_pem=`; the environment
variables apply to its helper subprocess when these arguments are omitted.

```sh
cargo run -p imajev-client --example paid_raw_call -- \
  --url http://localhost:8100/ --identity-pem "$paid_identity_pem" \
  "$paid_test_canister" getInferenceQuote artifacts/paid-prefix5/quote.args.bin
```

Use a dedicated local canister installed with this build and the real model.
Prepare its weights with `prepare_weight_cache.py`. Keep saved prefix queries
from the five-token prompt and export matching NPF1 packets using
`prepare_prefix_reuse.py`; packet manifests must bind to that exact saved prefix
report and each layer state. Old prefix27 packets are rejected.

```sh
.venv/bin/python -B scripts/prepare_fixed_prefix_states.py \
  --canister "$paid_test_canister" --wasm artifacts/paid-prefix5/build/full.wasm \
  --prefix artifacts/browser-prefix5-v1 --packets artifacts/paid-prefix5/packets \
  --directory artifacts/paid-prefix5/fixed-cache
.venv/bin/python -B scripts/register_common_update_prefix.py \
  --canister "$paid_test_canister" --wasm artifacts/paid-prefix5/build/full.wasm \
  --prefix artifacts/browser-prefix5-v1 --packets artifacts/paid-prefix5/packets \
  --directory artifacts/paid-prefix5/registration
.venv/bin/python -B scripts/configure_paid_poc.py \
  --canister "$paid_test_canister" --directory artifacts/paid-prefix5/config
```

Cache preparation tools explicitly use the local network and `imajev-local`
owner identity. The paid call helper uses the network and PEM selection above.
Cache preparation uses `prepareFixedPrefixCache`; registration uses
`installInferencePrefix`. Every asset is validated before owner updates. Cache
preparation is idempotent for identical entries. It prepares 24 Delta layers;
registration installs those and the eight Attention layers, covering all 32.

## Boundary, payment and refund proof

Install the owner-managed `examples/paid-inference-caller` relay on a dedicated
local caller canister and fund it with local cycles. The current proof takes an
explicit prepared model canister and relay. It never replaces a running module,
stops the shared network, or creates a model snapshot.

```sh
.venv/bin/python -B scripts/prove_paid_prefix5_local.py \
  --canister "$paid_test_canister" --relay "$paid_test_relay" \
  --wasm artifacts/paid-prefix5/build/full.wasm \
  --directory artifacts/paid-prefix5/proof --fault-recovery
```

The default fixture is generated from current BOOM #653 using `TextPreparer`.
Length stress fixtures preserve its prefix and chat tail. By default the proof
requires complete inference at 94, 95, 512, 513, 768 and 1024 total tokens, rejects 1025 before
payment, and verifies duplicate replay without a second fee. Accepted worker
failures do not count as successful boundary checks. Diagnostic mode additionally
injects a worker failure and a failed refund, retries the refund, checks repeated
retry/replay, and requires a fresh inference to return the same decision.

Reports remain `complete:false` if any required assertion fails. These synthetic
length fixtures test protocol bounds and payment behavior, not natural-language
accuracy at every possible length. The report does not claim independent query
logit parity or actual cycle burn unless those are measured separately.

```sh
.venv/bin/python -B scripts/measure_paid_prefix5.py \
  --proof artifacts/paid-prefix5/proof/report.json \
  --output artifacts/paid-prefix5/tariff.json
.venv/bin/python -B scripts/configure_paid_poc.py \
  --canister "$paid_test_canister" --config artifacts/paid-prefix5/tariff.json \
  --directory artifacts/paid-prefix5/config-measured
```

The estimate uses successful measurements for the same module and five-token
layout, with the existing 13-node execution coefficient and message overhead
floor. Storage remains operator-funded. Run additional short-length calibration
cases before adopting a tariff; a local execution estimate is not a production
subnet cycle measurement.

## Automated checks

```sh
.venv/bin/python -B -m unittest discover -s scripts -p 'test_paid_prefix5.py'
.venv/bin/python -B -m unittest discover -s scripts -p 'test_build_prefix_contract.py'
.venv/bin/python -B -m unittest discover -s scripts -p 'test_build_paid_contract.py'
.venv/bin/python -B -m unittest discover -s scripts -p 'test_prepare_text_limits.py'
cargo test -p imajev-runtime --lib \
  --features experimental-adaptive-token-tiles,experimental-delta-state-layout,experimental-blake3 \
  delta_hybrid::tests
cargo test -p imajev-runtime --lib \
  --features experimental-adaptive-token-tiles,experimental-delta-state-layout,experimental-blake3 \
  prefix_state_cache::tests
cargo test -p imajev-inference --features paid-update-diagnostics --lib
cargo test -p imajev-inference --features paid-update-diagnostics --test api_names
```

These commands verify admission/stream boundaries, 1019-token recurrence carry,
24-layer fixed-cache preparation and retry, and existing receipt/refund state
transitions. They do not replace the real-model local proof above. Production
publication is a separate operation.

The browser, unchunked and chunked optimized builders share
`build_paid_contract.py`. It copies the complete current paid source set,
including the stable receipt archive, and aligns the upgrade metadata footer
with the actual stable memory size. An unknown or ambiguous source layout
fails before compilation. The helper is included in each build's source hashes.
The builder unit tests exercise legacy footer fixtures and current source, unchanged
numerical code, missing archive detection and repeated application. Text boundary
tests tokenize complete prompts at 512, 513, 1024 and 1025 tokens.
