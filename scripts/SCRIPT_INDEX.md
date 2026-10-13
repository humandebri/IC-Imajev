# Script index

This index covers every publishable retained script/source file. **Current** means aligned with checked-in source contracts; it does not certify the live deployment. The browser entry point is `frontend/src/inference-client.ts`, and the paid entry point is the Candid API described in the [caller guide](../examples/paid-inference-caller/README.md). Historical Python drivers are not substitutes for those interfaces.

For a standard source build, use Cargo as described in the [root README](../README.md). The optimized builder uses public WAT and a Cargo-built patcher; browser asset export requires a matching retained prefix cache.

## Current utilities and maintained build entry points

| File | Target / data | Effects |
| --- | --- | --- |
| [check_publication.py](check_publication.py) | Local working files; optional reachable history | Read-only; redacted candidate output |
| [package_public_release.py](package_public_release.py) | Reviewed source and exactly verified browser assets | Writes an ignored history-free source tarball; no push or deployment |
| [script_usage.py](script_usage.py) | Historical CLI descriptions only | Read-only; imports no models or transports |
| [bootstrap.py](bootstrap.py) | Pinned upstream source | Network download; writes ignored vendor/artifacts |
| [acquire.py](acquire.py) | Pinned model metadata/weights | Network download; writes ignored checkpoints; verifies immutable lock |
| [inventory.py](inventory.py) | MODEL_LOCK and pinned model config | Writes ignored docs/capacity.json |
| [reference.py](reference.py) | Pinned official host model; Apple Silicon/MLX | Host inference; writes a fresh result; no canister calls |
| [export_full.py](export_full.py) | Pinned text model; separate original adapter | Writes native/INT8 packs and manifests |
| [export_projection.py](export_projection.py) | Pinned layer-0 projection primitive | Writes local primitive packs; not whole-model deployment |
| [export_readout.py](export_readout.py) | Pinned readout primitive | Writes local precision-variant pack; not deployment |
| [prepare_text.py](prepare_text.py) | Current five-token prompt contract | Writes prepared token IDs; no canister calls |
| [prepare_browser_examples.py](prepare_browser_examples.py) | Saved public BOOM fixtures/current tokenizer | Explicit regeneration uses pinned Python tokenizer; --check-content verifies saved compaction with standard Python; no refresh during ordinary build |
| [prepare_browser_prefix.py](prepare_browser_prefix.py) | Pinned browser module and matching five-token cache | Writes static browser assets; requires ignored pack/prefix evidence |
| [refresh_boom_neurons.py](refresh_boom_neurons.py) | Public BOOM indexed API | Read-only network fetch; writes the saved fixture |
| [configure_paid_poc.py](configure_paid_poc.py) | Explicit local canister; network local | Owner configuration update; test tariff; not public pricing |
| [prepare_fixed_prefix_states.py](prepare_fixed_prefix_states.py) | Explicit canister at localhost:8001 and matching Wasm/prefix | Owner cache updates; requires local keys and prepared data |
| [check_canister_api_exports.py](check_canister_api_exports.py) | Explicit Wasm file | Read-only export validation |
| [generate_wasm_candid.py](generate_wasm_candid.py) | Explicit Wasm file | Runs local Wasm in Node to extract Candid; writes output; no deployment |
| [build_optimized_inference.py](build_optimized_inference.py) | Public Rust/WAT/Cargo sources, Rust 1.97.1 | Writes raw/fully patched Wasm, actual Candid and hash report; never approves migration |
| [verify_optimized_paid.py](verify_optimized_paid.py) | Explicit loopback modules/relay and generated length fixtures | Runs paid comparisons and receipt/auth checks; never approves migration |
| [check_paid_instruction_evidence.py](check_paid_instruction_evidence.py) | Original paid result JSON and immutable checkpoint counter evidence | Rechecks numerical results, whole-message totals and conservative per-message budgets before migration proposals |
| [paid_instruction_accounting.py](paid_instruction_accounting.py) | Explicit dedicated local subnet checkpoints | Shared replica counter reader and conservative per-message accounting; no network calls |
| [paid_state_evidence.py](paid_state_evidence.py) | Immutable local Wasm heap checkpoints | Reads and verifies intermediate state evidence; model-bearing heap files stay in ignored local directories |
| [verify_paid_state_layout.py](verify_paid_state_layout.py) | Empty source-built diagnostic canister on explicit loopback IC | Seeds a known diagnostic fixture and compares the typed getter with the local heap decoder |
| [verify_optimized_paid_contract.py](verify_optimized_paid_contract.py) | Explicit dedicated loopback production candidate and cycles relay | Tests active upgrade rejection, live failure/refund, receipt scope/replay and safe upgrade; stops/starts/upgrades only the selected local canister; check mode revalidates saved evidence |
| [prepare_optimized_prefix.py](prepare_optimized_prefix.py) | Explicit loopback candidate, model manifest and native bridge | Computes a fresh prefix using canister queries; writes hash-verified states; never changes pins |
| [prepare_optimized_validation.py](prepare_optimized_validation.py) | Explicit dedicated loopback target, model and public saved prefix | Uploads/warms verification inputs; writes local report; never changes browser pins |

## Historical reproduction — explicit opt-in

`build_paid_token_chunks.py --historical` requires frozen runtime/compiler inputs and donor Wasm. Use `build_optimized_inference.py` for maintained public source builds.

Every CLI below stops **before optional dependency imports, artifact reads/writes or canister calls** unless `--historical` is present. `--help` without opt-in displays its historical scope without loading model dependencies. Use `--historical --help` for the original parser options in a prepared Python environment. Imported helper APIs retain their behavior; this is a CLI misuse guard, not an authorization or sandbox boundary.

```sh
python3 scripts/script_usage.py
python3 scripts/reproduce_proposal_assessment.py --help
python3 scripts/reproduce_proposal_assessment.py --historical prepare \
  --snapshot-archive <retained-snapshot-archive> --directory artifacts/new-reproduction
python3 scripts/reproduce_prefix_inference.py --historical --help
```

Historical child calls forward the flag only to known drivers inside this checkout. Original archived executables retain their original parsers. Frozen source-hash checks are still enforced: changing a CLI wrapper/helper does not authorize rewriting historical manifests or treating old results as new. Reproduce with original archived sources when a retained proof requires their exact bytes.

| File | Historical contract | Effects |
| --- | --- | --- |
| [build_paid_token_chunks.py](build_paid_token_chunks.py) | Frozen runtime/compiler and parent-Wasm build | Requires --historical; writes a fresh historical candidate |
| [reproduce_proposal_assessment.py](reproduce_proposal_assessment.py) | Preferred name for October 7, 2026 reassessment | May prepare prefixes and send historical local queries/codec updates |
| [reproduce_prefix_inference.py](reproduce_prefix_inference.py) | Preferred name for historical prefix/query graphs | Queries explicitly selected target; old graph/cache/module contract |
| [run_fresh_proposal_assessment.py](run_fresh_proposal_assessment.py) | Compatibility name: fresh means new queries on frozen modules | Same guarded behavior as reproduce_proposal_assessment.py |
| [run_prefix_canister.py](run_prefix_canister.py) | Compatibility name for historical graph driver | Same guarded behavior as reproduce_prefix_inference.py |
| [assess_proposal_range.py](assess_proposal_range.py) | Frozen 500–660 assessment and archived runner | Prepares inputs, executes archived queries or reads retained reports |
| [benchmark_proposal_assessment.py](benchmark_proposal_assessment.py) | Frozen 600–660 benchmark and local module/ID | Prepares inputs, sends local queries or scores retained outputs |
| [evaluate_prompt_accuracy.py](evaluate_prompt_accuracy.py) | Original/short-prompt comparison on a frozen module | Local queries plus historical prefix codec updates |
| [prepare_prefix_reuse.py](prepare_prefix_reuse.py) | Historical prefix codec contract | Owner updates through network local; explicit canister and expected module |
| [prepare_all_proposal_windows.py](prepare_all_proposal_windows.py) | Historical proposal window preparation | Reads retained archives and writes input/report files; no live inference |
| [prepare_all_proposal_query32.py](prepare_all_proposal_query32.py) | Historical prefix27/query32 input preparation | Reads retained archives and writes input/report files; no live inference |
| [prepare_all_proposal_compact86.py](prepare_all_proposal_compact86.py) | Historical prefix27/86-token input preparation | Reads retained archives and writes input/report files; no live inference |
| [build_paid_message_checkpoint.py](build_paid_message_checkpoint.py) | Archived paid candidate build | Uses fixed historical artifact destinations; local build only |

## Historical frozen builder names

Use `build_optimized_inference.py` for public source builds. All entries below require `--historical` and retain their frozen inputs and checks. None deploys a canister.

| File | Role |
| --- | --- |
| [reproduce_frozen_optimized.py](reproduce_frozen_optimized.py) | Preferred guarded frozen reproduction entry point. |
| [build_common_prefix27.py](build_common_prefix27.py) | Guarded compatibility name for frozen reproduction. |
| [build_latest_common_prefix27.py](build_latest_common_prefix27.py) | Guarded frozen implementation; not the public source builder. |

## Separate experiments and archived conformance

| File | Role |
| --- | --- |
| [export_merged_adapter.py](export_merged_adapter.py) | Separate merged-LoRA/requantized pack experiment; changes numerical computation and is not the default model pack. |
| [check_proposal_replacement.py](check_proposal_replacement.py) | Offline conformance comparison against an explicitly provided archived package/snapshot baseline; imports that baseline and writes a report. |

## Maintained code generation

These overwrite the named generated sources locally. They do not query or update a canister. Regenerated runtime bytes must match the intended source/reference before making numerical or performance claims.

| File | Role |
| --- | --- |
| [generate_int8_tile.py](generate_int8_tile.py) | Rewrites the checked-in block256 INT8 SIMD tile from its specification. |
| [generate_lane_pair.py](generate_lane_pair.py) | Rewrites the retained lane-pair kernel input; not a benchmark runner. |
| [generate_output_pairs.py](generate_output_pairs.py) | Rewrites output_pairs_simd.rs from the retained lane-pair kernel input. |
| [generate_output_pairs_shared.py](generate_output_pairs_shared.py) | Rewrites output_pairs_shared.rs from output_pairs_simd.rs. |
| [lane_pair_bench/src/kernel.rs](lane_pair_bench/src/kernel.rs) | Input consumed by the maintained SIMD generators; no benchmark Cargo project remains. |

## Imported / compiled support sources

These are dependencies rather than recommended Python commands. Transport helpers can have network effects when called by an authorized entry point; being a helper does not make their functions read-only. Rust CLI example sources are built through their configured Cargo examples.

| File | Role |
| --- | --- |
| [build_paid_contract.py](build_paid_contract.py) | Aligns archived sources with current paid receipt and upgrade contracts. |
| [build_prefix_contract.py](build_prefix_contract.py) | Aligns archived scheduler/carry sources with the current paid prefix. |
| [build_sequence_contract.py](build_sequence_contract.py) | Aligns archived sources with current sequence-position bounds. |
| [canister_api_names.py](canister_api_names.py) | Applies method names, removals and diagnostic feature gates. |
| [checkpoint.py](checkpoint.py) | Local safetensors reader; imported by pack exporters. |
| [evaluation_run_lock.py](evaluation_run_lock.py) | Serializes access to one retained evaluation output directory. |
| [paid_prefix.py](paid_prefix.py) | Current five-token prefix constants and validated saved-cache helpers. |
| [paid_update_transport.py](paid_update_transport.py) | Paid Candid/relay transport; can send calls when used by a CLI; network local. |
| [prepare_text_legacy.py](prepare_text_legacy.py) | Historical short-v2 prompt preparation; imported compatibility contract, not current browser input. |
| [proposal_snapshots.py](proposal_snapshots.py) | Reads and validates snapshots inside an explicitly selected archive. |
| [public_query_build.py](public_query_build.py) | Applies current public-query policy to archived optimized sources. |
| [repository_paths.py](repository_paths.py) | Validates external archive arguments before experiment output or transport creation. |
| [paid_raw_call.rs](paid_raw_call.rs) | Rust CLI example source for the native raw paid-call helper. |
| [paid_update_args.rs](paid_update_args.rs) | Rust CLI example source for paid Candid input/output encoding. |
| [prefix_state_args.rs](prefix_state_args.rs) | Rust CLI example source for prefix-state Candid encoding. |
| [prefix_state_cache.rs](prefix_state_cache.rs) | Rust module directly included by imajev-runtime. |
| [prefix_state_cache_canister.rs](prefix_state_cache_canister.rs) | Support source for specialized prefix-cache canister builds. |
| [query_attention_mlp_front.rs](query_attention_mlp_front.rs) | Support source used by specialized optimized canister builds. |

## Offline regression tests

Run with `PYTHONPATH=scripts:. .venv/bin/python -B -m unittest scripts.test_NAME`. These tests use local fixtures/mocks rather than live canister operations. Tokenizer tests additionally require pinned model metadata and upstream prompt code.

| File | Checks |
| --- | --- |
| [test_browser_examples.py](test_browser_examples.py) | Exact-value and rejection tests for built-in example compaction. |
| [test_build_paid_contract.py](test_build_paid_contract.py) | build paid contract |
| [test_build_prefix_contract.py](test_build_prefix_contract.py) | build prefix contract |
| [test_canister_api_names.py](test_canister_api_names.py) | canister api names |
| [test_journal.py](test_journal.py) | Client resume tests independent of model arithmetic and replica availability. |
| [test_paid_update_transport.py](test_paid_update_transport.py) | paid update transport |
| [test_optimized_build.py](test_optimized_build.py) | Pinned optimized build environment and inherited compiler override rejection. |
| [test_optimized_paid.py](test_optimized_paid.py) | Fresh receipt identities and rejection of checkpoint-only performance approval. |
| [test_paid_instruction_accounting.py](test_paid_instruction_accounting.py) | Replica counter precision, malformed protobuf rejection, unrelated execution and whole-message bounds. |
| [test_paid_instruction_evidence.py](test_paid_instruction_evidence.py) | Original evidence hashes, complete length coverage, numerical comparisons and instruction limits. |
| [test_paid_state_evidence.py](test_paid_state_evidence.py) | Heap overlay precedence, exact intermediate values, signed zero and malformed/ambiguous evidence rejection. |
| [test_paid_contract_evidence.py](test_paid_contract_evidence.py) | Original live contract responses, module identities, failed lifecycle actions and receipt scope. |
| [test_wasm_candid_metadata.py](test_wasm_candid_metadata.py) | Actual Candid extraction with existing Wasm custom sections. |
| [test_prepare_text_legacy.py](test_prepare_text_legacy.py) | Pinned-tokenizer regressions for the historical preparation contract. |
| [test_prepare_text_limits.py](test_prepare_text_limits.py) | prepare text limits |
| [test_proposal_assessment_runs.py](test_proposal_assessment_runs.py) | Exercise retries and report acceptance without calling a canister. |
| [test_proposal_benchmark_archive.py](test_proposal_benchmark_archive.py) | The 600–660 preparer reads and verifies the selected copied archive. |
| [test_proposal_contracts.py](test_proposal_contracts.py) | Evidence boundaries for the replacement snapshot and exact-encoding utilities. |
| [test_proposal_snapshot_archives.py](test_proposal_snapshot_archives.py) | Exercise archive selection without inference or canister access. |
| [test_public_query_build.py](test_public_query_build.py) | public query build |
| [test_publication.py](test_publication.py) | Exercise redaction, ignored credentials, deleted files and historical exposure. |
| [test_review_fixes.py](test_review_fixes.py) | review fixes |
| [test_script_usage.py](test_script_usage.py) | Historical CLI gates must stop before optional imports or experiment side effects. |

Publication bundle safety regression tests: [test_public_release.py](test_public_release.py).
