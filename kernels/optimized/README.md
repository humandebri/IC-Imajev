# Source-built optimized kernels

These 34 gzip-compressed WAT sources are retained from the verified local optimized-kernel experiments. `manifest.json` records their exact original SHA-256 and compiled function-body SHA-256. The original relative experiment path is provenance only, never a build input. The numerical backend in `crates/imajev-runtime/src/optimized` records its source origins separately; applicable Laya-derived arithmetic retains the MIT notice in `docs/licenses/Laya-MIT.txt`. Other project code is Apache-2.0; see `NOTICE.md`.

The patcher expands each public `.wat.gz` source in memory, checks its original uncompressed SHA-256, then compiles every WAT source, checks the standalone function's name and nine-I32 ABI, and requires the exact approved body hash. Sources omitting a memory declaration get a validation-only memory; function bytes do not change. `f32_output64.wat.gz` intentionally uses a generic original donor export name, explicitly mapped by the manifest. No donor Wasm is loaded.

The raw module must export exactly the 34 declared kernel targets. Every target must have a unique function index and the matching ABI. Extra/missing targets, aliases, incorrect source/body hashes or invalid Wasm abort without creating the final output. The patched module is a build candidate only; model-free synthetic probes do not replace local full-model numerical/performance gates.

Expanded `.wat` files are ignored. Gzip headers have fixed timestamps and no absolute paths. Compression is storage only: every build expands, checks the original source hash and recompiles all kernels.
