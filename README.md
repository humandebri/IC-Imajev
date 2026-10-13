# IC-Imajev

Public sources: [humandebri/ic-imajev](https://github.com/humandebri/ic-imajev). Download the matching [source release bundle](https://github.com/humandebri/ic-imajev/releases) for the pinned browser assets.

Run the pinned Imajev-4B **text decision model** on Internet Computer canisters. The model scores supplied choices and an unknown outcome using Qwen3.5-4B, a rank-64 LoRA adapter, and a dedicated decision readout. This implementation does not provide image inference or free-form text generation.

The Rust/Wasm runtime uses an INT8 backbone and F32 LoRA/readout. A Python client prepares inputs and carries intermediate state between ordinary queries. The React frontend prepares and executes anonymous queries in a Web Worker. A separate paid update API accepts cycles from calling canisters.

This is an experimental implementation. Standard Cargo and source-built optimized modules are separate build profiles. Historical measurements and the pinned browser release describe their original modules; a new module must pass the numerical and performance gates before migration.

## Public endpoints

| Resource | Address |
| --- | --- |
| Web playground | [imajev.kinic.xyz](https://imajev.kinic.xyz) |
| Inference canister | `xis3j-paaaa-aaaai-axumq-cai` |
| Canister details | [Internet Computer Dashboard](https://dashboard.internetcomputer.org/canister/xis3j-paaaa-aaaai-axumq-cai) |
| IC API gateway | `https://icp-api.io` |
| Source and issues | [humandebri/ic-imajev](https://github.com/humandebri/ic-imajev) |
| Source bundles | [GitHub releases](https://github.com/humandebri/ic-imajev/releases) |
| Browser API | [Public query API](docs/PUBLIC_QUERY_API.md) |

The canister is an inference API, not an HTML website. `frontend/src/inference-release.json` is the canonical canister/gateway/module binding used by the browser. Repository documentation changes do not upgrade that canister or deploy the web playground.

## Start here

| Goal | Entry point |
| --- | --- |
| Understand the source layout | [Documentation index](docs/README.md) |
| Build and test the Rust code | Commands below |
| Run or build the browser playground | [Frontend guide](frontend/README.md) |
| Integrate browser inference | [Public query API](docs/PUBLIC_QUERY_API.md) |
| Call paid inference from another canister | [Caller example](examples/paid-inference-caller/README.md) |
| Understand numerical contracts | [Runtime architecture](docs/runtime/ARCHITECTURE.md) and [inference-core](crates/inference-core/README.md) |
| Inspect historical results | [English evaluation report](docs/PROPOSAL_FILTER_REPORT_20261007.en.md) |

## Build and test

Rust and the `wasm32-unknown-unknown` target are required for a canister build. The Rust checks do not need model weights, a local IC network, or credentials.

```sh
cargo test --workspace --locked
rustup target add wasm32-unknown-unknown
cargo build --release --locked --target wasm32-unknown-unknown -p imajev-inference
cargo build --release --locked -p imajev-client -p imajev-runtime
cargo run --example export_did -p imajev-inference --locked > canisters/inference/inference.did
```

The default canister features include experimental paid inference and token tiling. Diagnostic methods require `paid-update-diagnostics`. See the [API guide](docs/CANISTER_API_NAMES.md) for service definitions and the difference between standard and optimized releases.

## Model preparation and host reference

[MODEL_LOCK.json](MODEL_LOCK.json) pins upstream revisions and SHA-256 hashes. Python dependencies are pinned in [requirements.lock](requirements.lock); the MLX host reference was evaluated on Apple Silicon with Python 3.12. This is not a claim of host-reference portability to other platforms.

The following commands download pinned upstream code and approximately 10 GB of model data into ignored directories. They require network access and `uv`.

```sh
python3 scripts/bootstrap.py
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python 3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements.lock
python3 scripts/acquire.py --weights
.venv/bin/python scripts/inventory.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/reference.py \
  --first-only --output artifacts/reference-first.json
```

Acquisition refuses to replace an existing lock with different content. Reference outputs use fresh destinations; existing outputs are not overwritten. [benchmarks/cases.json](benchmarks/cases.json) contains retained input fixtures. Export helpers and their dependencies are listed in the [scripts guide](scripts/README.md).

For a history-free source package with the pinned browser assets, follow [Public source releases](docs/RELEASING.md). The package builds without local model/cache archives. The source-only optimized builder includes the 34 verified WAT kernels and a Cargo-built patcher.

## Execution and reproducibility

See [CI](docs/CI.md) for the checks run on pull requests and `main`, and how to run them locally.

Canisters hold immutable weights and prepared prefix caches. Ordinary queries return intermediate carries to the client, which binds them to model, pack, input, progress, and checksum. Checksums do not replace response authentication. Shared prefixes reuse computation for identical initial tokens; they do not summarize inputs.

The browser release uses a five-token prefix and a total budget of 6–96 tokens, including prompt formatting and choices. The pinned release and accepted query plans live in `frontend/src/inference-release.json` and `frontend/src/query-profiles.json`. Paid inference has its own limits and pricing, which callers obtain through `getInferenceQuote`.

Historical local evaluation on October 7, 2026 grouped 50 eligible BOOM proposals into 18 unique inputs. These used 32 or 50 inference queries each, totaling 774 inference queries, with 990 additional prefix/codec preparation queries. They reproduced the previous adopted INT8 logits bit for bit. These are historical local measurements, not current deployment latency, general model accuracy, or equivalence to the unquantized model. See the [evaluation report](docs/PROPOSAL_FILTER_REPORT_20261007.en.md) for inputs, scope and limitations.

Build an optimized candidate with `python3 scripts/build_optimized_inference.py --directory .cache/new-optimized-build`. Rust 1.97.1 and Cargo.lock are pinned. This command needs no retained Wasm, model weights, compiler archive or Python virtual environment. It produces separate raw/optimized modules, Candid extracted from the optimized module, and a source/kernel/toolchain hash report. The existing frontend remains pinned to its previously measured module until a candidate passes all migration gates. See [the release guide](docs/RELEASING.md).

## Local IC development

[icp.yaml](icp.yaml) configures a project-local network on port 8001. Check that the port is available and inspect the actual endpoint before creating dedicated test canisters. Use the `icp` CLI; keep local identities, exports and snapshots outside Git. Deployment and cache preparation are explicit operations, separate from source builds. A canister upgrade changes the module identity and may require rebuilding heap caches and updating frontend release pins.

## License and contributions

The project is licensed under [Apache-2.0](LICENSE). [NOTICE.md](NOTICE.md) records upstream model, source and fixture attribution; the retained [MIT notice](docs/licenses/Laya-MIT.txt) must accompany applicable material. Model weights are downloaded separately. See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and [SECURITY.md](SECURITY.md) for reporting guidance.
