# Public source releases

The public source repository is [humandebri/IC-Imajev-public](https://github.com/humandebri/IC-Imajev-public), initialized from a reviewed snapshot without development Git history. Download the matching source bundle from its [releases](https://github.com/humandebri/IC-Imajev-public/releases); bundles include the generated browser assets omitted from ordinary Git tracking. The original `humandebri/IC-Imajev` development repository is retained privately. Before the October 13, 2026 cutover it was public, and its history contained deleted operational notes and machine-specific paths. Making it private does not revoke copies already obtained by others. Do not copy its existing refs into this source repository. Existing email attribution remains in the original history; source packages contain no commit metadata. See the [history cleanup record](GIT_HISTORY_CLEANUP.md).

## Prepare a release

From a reviewed development checkout with Node.js 24 and the pinned browser assets already generated:

```sh
python3 scripts/check_publication.py
node frontend/scripts/check-release-assets.mjs
python3 scripts/package_public_release.py
```

The last command creates `.cache/public-release/ic-imajev-source.tar.gz` and prints its SHA-256. It refuses to overwrite an existing package; choose a new ignored output with `--output` for a later version. The deterministic package contains current publishable files plus exactly 70 checksum-verified browser assets and the upstream tokenizer license. It excludes `.git`, deleted experiments, owner credentials, model weights, frozen compiler inputs and arbitrary files in ignored directories. Pattern checks also apply to the selected asset bytes. `PUBLIC_SOURCE_MANIFEST.json` inside the package records each included file's digest, size and mode; it excludes itself.

Extract into an empty directory, check the manifest and review the result before uploading. The packager does not create a repository, push, change refs or publish anything. To start a public Git repository, initialize one from the extracted tree and give it a new remote. Never copy the development `.git` directory. Keep the historical experiment archive private unless separately reviewed.

## Build from the extracted release

```sh
tar -xzf ic-imajev-source.tar.gz
cd ic-imajev
cargo test --workspace --locked
cargo test --workspace --locked --all-features
cargo build --locked --release --target wasm32-unknown-unknown -p imajev-inference
cd frontend
npm ci
npm run build
```

Install Rust's `wasm32-unknown-unknown` target first. Browser builds require Node.js 24, npm and standard Python 3; NumPy, model weights and the development `.venv` are unnecessary for this route. Build hooks verify tokenizer/readout hashes against `MODEL_LOCK.json`, both prefix asset copies against the release manifest hash, and module/model/pack/prompt identities. They regenerate the public Candid and verify saved example compaction and actual browser token counts without fetching new inference or governance data. Missing or mismatched assets cause a failure before Vite builds. Extract the matching source release instead of substituting assets from another module.

A plain Git clone has source but no generated browser assets. Supply the `frontend/public/tokenizer/` and `frontend/public/inference/` directories from its matching source release package. Maintainers can instead explicitly regenerate them with `npm run prepare:tokenizer` and `npm run prepare:inference`; that route requires the original ignored tokenizer/readout files and prefix cache documented in the frontend guide. Asset regeneration is not part of ordinary builds.

## Supported reproduction scope

The Cargo release command rebuilds the checked-in canister implementation. The browser package reproduces the pinned static application and prefix inputs. Neither claims to regenerate the module currently pinned in `frontend/src/inference-release.json`: the browser checks that remote module when used.

## Source-only optimized candidates

`icp build imajev-inference`, the optimized builder below, and the paid token-chunk builder embed the full Candid extracted from their actual Wasm as public `candid:service` metadata. They require Node.js and Python 3. The embedder preserves every other Wasm section byte for byte and replaces any existing public/private Candid section. The browser query subset is not used for this metadata. Embedding occurs before the final module hash is recorded; existing frontend module pins must be updated through the usual migration gates. A direct `cargo build` alone does not embed this metadata.

After deploying the resulting module, anyone can retrieve its interface with `icp canister metadata <canister-id> candid:service -n ic --identity anonymous`, including Dashboard clients. Adding the build step does not change an already deployed canister.

```sh
python3 scripts/build_optimized_inference.py --directory .cache/new-optimized-build
# Optional model-free verification module; never use this module for migration:
python3 scripts/build_optimized_inference.py --directory .cache/new-kernel-check --kernel-contract-tests
```

The builder fixes Rust 1.97.1, locked dependencies, SIMD and the explicit `optimized-release` feature. It compiles all 34 public WAT kernels on every invocation and builds the patcher through Cargo. Kernel source/body hashes, names, ABI, export coverage and final Wasm validity are mandatory. Existing Wasm, frozen compiler libraries and executable tools are not build inputs. The standard Cargo output is `raw.wasm`; its placeholders are deliberately unusable for inference. Only a fully patched `optimized.wasm` is emitted as a candidate. `service.did` comes from that actual module. `report.json` records relative source paths, toolchain/features, kernel hashes and final hashes. Inherited Rust flags/wrappers are cleared, source/Cargo paths are remapped, and output directories must be new. `--offline` uses already downloaded ordinary Cargo dependencies.

A build report is not migration approval. All browser suffixes 1–91 and paid short/96/128/256/512/1024-token cases must match the original state/payload/logits bit for bit under identical model/cache conditions, stay within existing message and communication limits, and use no more than 105% of baseline instructions per input. Synthetic probes cover 704 arithmetic, predicate, RMS and Attention boundary cases and patcher tests cover malformed modules, ABI/name/hash mismatches, aliases and duplicate patching. Full-model tests use private local data only; source builds never depend on those inputs.

`prepare_optimized_validation.py` and `frontend/scripts/verify-optimized-query-release.mjs` operate only on explicit loopback canisters. Their outputs are local validation evidence, not replacement frontend assets. A new release must generate prefix assets and query measurements for the new module together; never relabel old measurements with a new module hash. Keep the current frontend pins and rollback inputs until all gates pass. This task does not upgrade public canisters, deploy sites or publish source.

## Historical reproduction

`reproduce_frozen_optimized.py --historical` retains the old verified frozen-input chain for historical investigations. Both compatibility names `build_common_prefix27.py` and `build_latest_common_prefix27.py` require the same opt-in. Its archived compiler and parent-Wasm extraction remain outside the public source build. Model weights, internal logs, frozen compiler libraries, historical experiments and Git history are excluded from the source package.
