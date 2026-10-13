# Runtime architecture and numerical contracts

| Location | Responsibility |
| --- | --- |
| `crates/inference-core/` | Model-independent F32 projection primitives, scalar references, BF16 and block256 codecs |
| `crates/imajev-runtime/` | INT8 kernels, prepared weights, LoRA, Attention, DeltaNet, MLP, frame validation and dispatch |
| `canisters/inference/` | Authorization, model preparation, stable/heap state, IC methods, paid jobs and receipts |
| `crates/imajev-client/` | Native bridges, transport and Candid input/output helpers |
| `client/` | Python model graphs, client-held carry, budgets and journals |
| `frontend/` | Browser tokenizer, query planning, verified prefix assets and response handling |

The dependency direction is `canisters/inference → imajev-runtime → inference-core`. `inference-core` has no external crate dependencies and does not reference model manifests, tensor names or IC APIs. It targets native execution and Wasm SIMD128. Model-specific raw SIMD calls are unsafe; callers must satisfy the documented shape and memory bounds.

F32 projection preserves each dot product's column order and multiply-then-add behavior while sharing loads between independent token lanes. BF16 packing stores the upper 16 bits; callers must verify BF16 representability before claiming a lossless round trip. The block256 codec validates finite values and canonical payloads and preserves signed zero. Discard partially appended output when encoding fails.

The retained `MAX_FLOATS=900000` and 2 MB codec frame budget reflect this implementation's limits. Matrix operations validate shapes, while callers remain responsible for input finiteness checks. INT8 quantization changes numerical computation; transport compression and reuse must be compared against the adopted INT8 reference rather than described as identity with the unquantized model.

Several experimental features and direct Rust source references still depend on files under `scripts/`. Those dependencies remain in the public tree. Frozen Wasm and old source hashes belong to their historical experiments and must not be rewritten as if current sources produced them.

A generic multi-model runtime would additionally require a second model, model description, loaders, scheduling, failure handling and comparable Wasm measurements. GGUF loading, a generic tokenizer and free-form generation are not implemented here.

For source validation, run `cargo test --workspace --locked` and the Wasm build documented in the [repository README](../../README.md). Numerical changes also need targeted comparisons with the appropriate reference; changing crate boundaries can change final Wasm performance even when outputs remain identical.
