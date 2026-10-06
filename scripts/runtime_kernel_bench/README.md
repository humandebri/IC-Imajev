# Runtime kernel comparison harness

Compare the pinned onicai GGML Wasm Q8_0 dot with Imajev prepared projection on synthetic integer inputs. This is a kernel diagnostic, not a model benchmark. Results and limitations: [KERNEL_BENCHMARK.md](../../docs/runtime/comparison/KERNEL_BENCHMARK.md).

Prerequisites: Rust with wasm32-unknown-unknown, Cargo offline dependencies, Apple clang with wasm SIMD support, rust-lld, Node, and the existing Imajev WAT patcher. The build reads hash-checked public fork sources from `artifacts/runtime-comparison/fork/`, using the URLs and hashes in `docs/runtime/comparison/fork-sources.json`. Retrieve those exact sources before building. Generated GGML code retains the fork MIT license alongside it.

The build also requires the existing kernel WAT files at `artifacts/prefix_codec/full-wat/reuse.wat`, `artifacts/s1_address_reuse/build/kernel.wat`, and `artifacts/s1_wide/build128/kernel.wat`, plus `artifacts/wasm-audit-target/release/imajev-wasm-patch`. It freezes relevant source and validates source hashes after building and measuring. Output directories must be new and inside this repository.

From the repository root:

```sh
python3 scripts/benchmark_runtime_kernels.py --local-ic --directory artifacts/runtime-comparison/new-build
```

This performs the Node byte equality and independent scalar checks before IC use. To measure IC instructions, start the selected loopback local IC network using the project's local-network workflow, then run:

```sh
python3 scripts/measure_runtime_kernels_ic.py --build artifacts/runtime-comparison/new-build --directory artifacts/runtime-comparison/new-proof --identity imajev-local
```

The IC runner requires `icp`, checks for a loopback endpoint, creates two detached diagnostic canisters, installs only those, makes 20 preparation updates and 60 ordinary queries, and stops/deletes both in cleanup. The raw diagnostic exports have no production authorization or persistence contract; use only the local runner. It records created IDs before proceeding so interrupted cleanup can be recovered from the report. An interruption that kills the process may require manually stopping/deleting those recorded IDs.

The measured fixture uses 512 output rows, token counts 1/7/32/87/132, and input columns 256/2560. Preparation and response hashing are outside the timed interval. GGML receives preallocated outputs; Imajev uses its public projection API including validation and allocation. See the report for all samples, module hashes and output proofs.
