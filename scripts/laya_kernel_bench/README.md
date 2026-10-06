# Laya prepared projection comparison

This diagnostic compares the current IC-Laya integer dot and F32 scale/bias writeback functions with Imajev's experimental full-column token-scale kernel. It does **not** compare the adopted block256 S1 kernel or full runtimes.

The runner reads the Laya checkout without modifying it. It verifies complete tensor SHA256 values against the real INT8 manifest, then selects the first 512 output rows of `encoder.0.qkv.weight` and `encoder.0.wo.weight`. Inputs are deterministic synthetic signed integers with nonuniform positive token scales. These are real weights, not captured model activations.

Both paths calculate full-column I32 sums, cast to F32, multiply by token scale, then weight scale. The baseline uses the extracted current Wasm `dot_tile` and `q8_store_tile` with no bias; a diagnostic wrapper preserves its 64/32/16/8/4/2/1 token and 16-output tile dispatch. Output requantization and group scratch assembly from the full Q8 linear API are excluded. The candidate uses `int8_token_kernel::project` without changing production code, and zero-pads 2624 columns to 2816. Its input validation, allocation, finite checks and extra padded arithmetic are in the measured span. Both paths allocate outputs and receive the same final finite check. This is not an equal-overhead microbenchmark.

Build and validate from the IC-Imajev root, using a new output directory:

```sh
python3 scripts/benchmark_laya_kernels.py --directory artifacts/runtime-comparison/new-laya-build
```

Prerequisites: Cargo offline dependencies, wasm32-unknown-unknown, and Node. `--laya-root` selects another checkout. Only the two tensor slices are copied; no full model download is needed. Generated source retains the Laya MIT license. The build freezes source, records provenance, and asserts source stability through compilation and Node validation. Node checks all output bytes against both backends and an independent scalar integer reference with explicit F32 rounding.

For IC measurements, use the already selected loopback local network:

```sh
python3 scripts/measure_laya_kernels_ic.py --build artifacts/runtime-comparison/new-laya-build --directory artifacts/runtime-comparison/new-laya-proof
```

The runner validates the Node proof and module hash, creates one detached diagnostic canister, prepares each shape, and alternates both paths for 60 ordinary queries. Unique query arguments avoid cache reuse. Cleanup stops and deletes only the created canister. Abrupt process termination can require manual cleanup using the recorded ID. These raw exports are for local diagnostics and have no production access-control or persistence contract.

Preparation of both input representations and padded weights is outside the measurement. IC replies, digest calculations and network waits are also excluded. Node wall time is a separate metric. The benchmark does not measure quantization, re-quantization, decision quality, full-model memory, or model loading cost.
