# Source-built optimized inference

The optimized profile builds from public Rust and 34 hash-verified WAT sources stored as deterministic `.wat.gz` inputs. It preserves the current source API, authorization, paid receipts/refunds and upgrade implementation. Only the numerical backend, lossless predicates and prepared prefix cache receive the verified optimizations. Current token/shape validation remains in force. The historical numerical origins and applicable Apache-2.0/MIT notices are recorded in `crates/imajev-runtime/src/optimized/PROVENANCE.json`, `kernels/optimized/manifest.json` and `NOTICE.md`.

```sh
python3 scripts/build_optimized_inference.py --directory .cache/new-source-build
python3 scripts/build_optimized_inference.py --directory .cache/new-kernel-verification --kernel-contract-tests
```

Both commands require Rust 1.97.1 and a fresh output directory. The production command selects `optimized-release` explicitly, with no default features; all-features tests are a separate check. Cargo.lock, source hashes and SIMD/path flags identify the configuration. The builder writes `raw.wasm`, fully patched `optimized.wasm`, its extracted `service.did`, compiler diagnostics and `report.json`. The raw placeholders must never be deployed. The verification command adds non-Candid arithmetic probes and marks its output as verification-only.

Every kernel is compiled from WAT during the build. Exact source/body hashes, nine-I32 function types, unique exports and complete 34-function coverage are required. Extra/missing kernels, aliases, already-patched inputs, invalid modules and incompatible donors are rejected. No function is extracted from retained Wasm and no precompiled patcher is used. Reports use relative source paths, and compiler paths are remapped. Public source packages omit model weights, credentials, internal logs, archived compiler libraries, experiments and Git history.

## Verification record

Before the compiler/profile override hardening, two empty public-source extractions at different absolute paths produced identical production Wasm, Candid and JSON build reports for module `47ef4176183453e9894ff24aa7ea21699263d3965b0837d7aaa6dc232279e0d0`. The compiler and ordinary locked Cargo dependencies were already installed; no development virtual environment, ignored artifacts/vendor directories or existing Wasm were present in either source tree. The report includes the external Rust prefix-cache helper as a hashed source input.

After compiler/profile hardening and SIMD finite validation of restored Delta state, two fresh public-source extractions again produced identical production Wasm, Candid and JSON build reports for module `ebbd809fba251071cd4dbf9619cf83e5c63693cddea71ce113c210900f5f451c`. Both builds used only the public source tree and installed locked Cargo/toolchain dependencies. Model-specific measurements for earlier module hashes do not validate this candidate.

Embedding the extracted public Candid metadata produced final module `a1ad5570b8ee70e65cdc26430341f33a3ed10af9a80808b5deb4c6a69b056ee7`. Two new empty source extractions at different absolute paths reproduced this Wasm, Candid and JSON report identically. Numerical measurements must use this final hash; results for the preceding module are retained only as development evidence.

Dedicated local paid measurements of this final module against reference `8fd78b39b4ab626b4623826c162275737d875b7df25341d1aa2852d6e50a546c` passed all six lengths. Every layer's hidden/state hashes, final hidden F32 bits and decision numerical bits matched exactly. Whole-message instruction totals and conservative individual-message bounds were rechecked from the original replica/heap evidence. The 512- and 1024-token runs included checkpoint-aborted worker retries, verified against their original self-call job/stage requests.

| Total tokens | Candidate/reference instructions | Conservative message bound (billions) |
| --- | --- | --- |
| 9 | 104.849122% | 12.885795351 |
| 96 | 101.043462% | 31.683841660 |
| 128 | 101.151749% | 32.578174469 |
| 256 | 101.615457% | 32.948530562 |
| 512 | 102.663717% | 33.283206364 |
| 1024 | 104.702585% | 35.013674327 |

On the same production module, active inference rejected a direct management upgrade. Stopping the dedicated local canister caused a real worker failure and full-fee cycles refund; caller scope, refund retry and replay checks passed. A subsequent safe upgrade preserved configuration and both completed/failed receipts. Anonymous retrieval of `candid:service` matched the extracted 20-method interface. These checks used the production module, with no diagnostic Candid methods.

Actual patched verification Wasm passed 480 INT8, 60 F32, seven Delta, 110 predicate/RMS, 35 elementwise and 12 Attention-score cases against scalar references. Comparisons use F32 bits and include zero/signed/extreme values, output seeding, tile boundaries/tails, Inf/NaN and SIMD scan boundaries. Workspace tests pass in ordinary and all-feature configurations; Python regression tests, Candid/export contracts and public browser asset/profile checks also pass.

The first source candidate matched every intermediate state, final payload and logits for browser suffix 1, but used 105.9231% of reference instructions (6,022,705,345 versus 5,685,922,411). It was rejected. Missing SIMD predicates and ordered RMS optimization were then integrated; that rejected candidate is never migration evidence for the newer module. Full-model measurements are generated on a dedicated loopback IC, with separate reference/candidate canisters and matching 4GiB memory limits, model, 721 weight caches and 24 prepared Delta prefixes. Validation data remains local and is not a build input.

**A validated local migration candidate is prepared.** For the final module above, all browser suffixes 1–91 passed bit-exact intermediate payload, final payload and logits comparisons against the reference. The largest candidate/reference total-instruction ratio was 101.545604%; the largest individual query used 3,944,784,980 instructions, below the 4-billion limit. Maximum request/reply sizes were 1,753,265 and 1,635,440 bytes, within the existing communication budgets. The six paid cases and production refund/authorization/upgrade checks described above also passed.

Fresh prefix assets were generated from the final production module and matched the operands held constant during comparison. The local proposal contains the candidate Wasm, extracted Candid, build report, new module/manifest pins, original candidate measurements and derived query profiles, plus previous frontend pins for rollback. Its migration report binds the independently revalidated query, paid and contract evidence by hash. Build reports still deliberately keep `release_eligible: false`: a successful build alone cannot approve migration. The existing frontend pins remain unchanged; no public canister upgrade, site deployment or external publication was performed. Model-specific replica and heap evidence remains in ignored local storage and is excluded from public source packages.

## Local comparison tools

`prepare_optimized_validation.py` takes explicit loopback URL, canister/module, model manifest/pack, native bridge/helper, dedicated project root and cache reference. It verifies module/model identities, warms missing weight caches and prepares the saved five-token prefix. `--install-paid-prefixes` also installs the 32 paid-prefix operands. It reconstructs exact saved BF16/F32 words rather than running host model inference. The only signing identity used is the explicitly supplied local validation identity.

Build the validation helpers from the current source before using them:

```sh
cargo build --locked -p imajev-client --example prefix_state_args --features imajev-runtime/experimental-prefix-hybrid
cargo build --locked -p imajev-client --example paid_update_args --example paid_raw_call
```

These helpers are native local validation programs, separate from the pinned production Wasm feature set.

`frontend/scripts/verify-optimized-query-release.mjs` compares all browser suffixes, recording input/state/logit hashes, request/reply sizes and instruction counts. Optional `--suffixes` performs a partial check without approving the full gate. Optional `--concurrency` overlaps independent, read-only query graphs; it never changes module/cache state. Any failed condition rejects the candidate and retains its evidence.

`frontend/scripts/prepare-optimized-paid-fixture.mjs` emits deterministic short and boundary-length fixtures. `verify_optimized_paid.py` compares dedicated loopback paid targets through an explicitly selected cycles relay, checks worker budgets, exact numerical bits, fees, receipt replay/scope and owner/self authorization. Its default request uses the current ABI. `--historical-reference-request` enables the former model/version fields only for the hash-pinned old reference module. This compatibility encoding does not change the candidate's public API. Paid comparison reports cannot approve migration while intermediate-state, live refund or upgrade gates remain incomplete.

These tools do not upgrade public canisters, deploy the frontend or publish a repository. Frozen reproduction has a separate entry point, `reproduce_frozen_optimized.py --historical`.

Each paid validation run generates a fresh request ID namespace and checks for an existing caller-scoped receipt before testing admission. Replays reuse that ID only inside the same run. Worker instruction counters are diagnostic checkpoints: they omit reply epilogues, admission and callback execution. Their sums and 105% comparison are recorded under explicit `worker_checkpoint` fields, not as total instructions.

For whole-message accounting, pass `--checkpoint-state-dir` with the explicitly selected dedicated local subnet's state directory. The validator retains immutable replica checkpoint metadata before and after each inference and reads the replica's cumulative execution counter. It checks module identity, canister versions and local request/callback counts, rejecting unrelated execution. The total includes admission, workers and callbacks. Subtracting all worker checkpoints from that total gives a conservative upper bound for the omitted part of any individual message; adding that remainder to the largest worker checkpoint must stay within the existing [40-billion IC update instruction limit](https://docs.internetcomputer.org/references/resource-limits/). The worker scheduler's 30-billion threshold is checked before starting another stage and is not a bound on the complete message. Every input must also remain within 105% of the reference total. The checkpoint wait defaults to 180 seconds and can be set explicitly with `--checkpoint-timeout`.

Only a complete six-length run with these measurements can set `paid_instruction_gate_passed` and `whole_message_accounting_complete`. Without the checkpoint option, both remain false. `python3 scripts/check_paid_instruction_evidence.py --report <report.json>` rechecks the original checkpoint bytes and paid response files, their hashes, numerical comparisons and budgets. Migration proposal preparation runs this checker before writing output. Replica metadata and model-specific results remain local validation files, outside the public source package. Intermediate-state, refund and upgrade gates remain separate requirements.

Evidence revalidation also requires positive integer request/reply sizes below 1,990,000 bytes; missing sizes or boundary/over-budget values reject the report. Heap retention uses hard links when possible and copies identical bytes on a cross-volume link error, while retaining the same checksum and decoder checks. Other link errors remain failures. Contract verification owns its inference helper process session: cleanup terminates and reaps helpers and descendants, then joins the verification thread before restarting the local canister or recording the evidence inventory.

The verification-only `--paid-contract-tests` builder enables existing owner fault/upgrade probes, extracts the actual 28-method Candid and marks the module as verification-only. The production command continues to omit those probes. Synthetic receipt metadata can exercise local upgrade guards and real cycles-deposit refund transitions without model weights.

Checkpoint construction can abort and retry an executing self-call. The accounting tool captures the original pending request, including caller, callback, method and Candid job/stage arguments, and permits extra local-message starts only when these immutable records prove the retry. Committed canister versions and the complete execution counter must still match the expected worker chain. Missing or unrelated retry evidence rejects the run.

For intermediate paid states, first use `verify_paid_state_layout.py` on an empty diagnostic canister built from the same paid-state sources and compiler as the production candidate. Its existing owner probe seeds 32 known hidden hashes, 32 state hashes and 2,560 exact F32 values. The typed Candid getter must match the heap decoder, including signed zero. Pass this conformance report and the production build report to `verify_optimized_paid.py` with `--state-layout-evidence`, `--candidate-build-report`, `--capture-paid-state` and `--checkpoint-state-dir`. The validator then compares all layer hashes and final hidden F32 bits on the actual reference and production modules. It checks immutable heap files, module identity and original hashes again before approving a proposal. Heap files contain model weights and must remain in ignored local storage; they are never public source inputs.

After the numerical run, `verify_optimized_paid_contract.py` uses the explicit dedicated local production candidate and relay to test an active inference upgrade rejection, an actual stopped-worker failure and full-fee refund, caller-scoped receipts, refund retry, replay and receipt/config preservation across a safe upgrade. It uploads the selected Wasm in chunks and submits `install_chunked_code` directly: the CLI's ordinary install path stops the canister first, so it cannot test an active pre-upgrade guard. All management calls use an explicit local effective canister ID. The verifier checks the loaded module before and after lifecycle operations and saves the original response bytes and file hashes. `check --report <contract.json> --paid-report <paid.json>` rechecks this evidence and its binding to the numerical report. A successful upgrade clears heap caches: prepare the candidate again before browser measurements. Proposal generation accepts the separate contract report only after both evidence checkers pass; the numerical report itself retains its separate contract requirement.

The builder now selects the installed Rust 1.97.1 compiler/Cargo explicitly, clears inherited compiler and profile overrides, and sets release profile options through Cargo command-line configuration. This changes the candidate module identity; older measurements remain bound to their original module. Public information checks expand bounded WAT gzip sources before scanning. Release proposals revalidate every original reference/candidate pair against the combined report hashes and numerical/budget conditions before writing files. Frozen paid builds require `--historical`, including calls through compatibility wrappers.

The native patcher is built with the pinned compiler's explicit host target, overriding inherited Cargo target settings, and is executed from that target's output directory. Before writing a migration proposal, the generator also verifies the candidate's manifest checksum, module/model metadata, and all 32 asset checksums and sizes in both browser prefix URL layouts. Missing or altered prefix files reject the proposal.
