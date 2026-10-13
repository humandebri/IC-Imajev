# Scripts: choose an entry point

Use the [complete script index](SCRIPT_INDEX.md) to check each file's target, prerequisites and side effects. It covers current utilities, historical reproduction, compatibility names, code generators, imported/compiled helpers and tests. A historical filename or the word “fresh” does not identify a current public-canister client.

## Current use

| Goal | Entry point |
| --- | --- |
| Build standard source | `cargo build` / `cargo test` from the [root guide](../README.md) |
| Use the public browser model | `frontend/src/inference-client.ts`; [browser API](../docs/PUBLIC_QUERY_API.md) |
| Call paid inference | [Caller example and Candid API](../examples/paid-inference-caller/README.md) |
| Check publication files/history | `check_publication.py` |
| Package history-free source and pinned browser assets | `package_public_release.py`; [release guide](../docs/RELEASING.md) |
| Download pinned source/model files | `bootstrap.py`, `acquire.py` |
| Prepare current prompt IDs | `prepare_text.py` |
| Export pinned browser assets/examples | `prepare_browser_prefix.py`, `prepare_browser_examples.py` |
| Build an optimized candidate from public source | **`build_optimized_inference.py`** |
| Configure local test pricing / prepare local prefix cache | `configure_paid_poc.py`, `prepare_fixed_prefix_states.py`; explicit dedicated local target required |

“Current” refers to checked-in source contracts. It does not assert that the live canister matches them. Optimized builds and browser asset export still require the retained data documented in the [frontend guide](../frontend/README.md). Builds do not deploy a canister.

## Historical reproduction

Prefer the names **`reproduce_proposal_assessment.py`** and **`reproduce_prefix_inference.py`**. The former reproduces the October 7, 2026 frozen-module assessment; “fresh” in its old name means new queries, not today's deployment. The latter runs old prefix/query graph contracts, not the pinned browser API.

```sh
python3 scripts/script_usage.py
python3 scripts/reproduce_proposal_assessment.py --help
python3 scripts/reproduce_prefix_inference.py --help
```

Historical CLIs, including their old names and preparation/benchmark dependencies, require **`--historical`** before they load optional dependencies or perform experiment work. To see original parser options with dependencies installed:

```sh
python3 scripts/reproduce_proposal_assessment.py --historical --help
python3 scripts/reproduce_prefix_inference.py --historical --help
```

Verify the original local target, module, prompt layout, cache and frozen data before executing. Historical preparation may send owner updates; historical inference may query fixed old local canisters. The gate is a misuse guard for CLI execution, not a replacement for authorization or module verification.

Use `build_optimized_inference.py --directory <new-output>` for public source builds. `reproduce_frozen_optimized.py`, `build_common_prefix27.py` and `build_latest_common_prefix27.py` retain the frozen-input implementation; all require `--historical`.

## Helpers and validation

Files listed as support sources in the index are imported or compiled dependencies, not recommended standalone commands. Four maintained generators rewrite local SIMD sources; `lane_pair_bench/src/kernel.rs` is their input, not a benchmark project. Separate merged-adapter experiments change numerical computation and are not the default model pack.

Retained offline tests use fixtures and mocks; tokenizer checks require pinned data. `test_script_usage.py` verifies that historical commands stop before imports and side effects, that help works without model dependencies, and that authorized historical child/fallback commands retain explicit opt-in.

Frozen source-hash checks remain enforced. Use original [archived sources](../docs/ARCHIVE.md) when a historical proof requires exact old bytes; do not rewrite frozen manifests to accept edited entry points. Deleted experiment files remain outside the public source tree.
