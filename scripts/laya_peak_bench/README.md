# Laya quantization peak scan diagnostic

Ports only the unsigned absolute-F32-bit maximum scan from Imajev's `quantize_simd.rs` to a frozen copy of Laya's `int8_quant.rs`. Laya's scale floor (`f32::MIN_POSITIVE`), zero scale, F32 division, half-away-from-zero rounding, clamp and output packing remain unchanged. This does not introduce Imajev's ties-to-even quantization or block256 scales.

The original Laya SIMD row function is the control. Both copies use the same small diagnostic `candle_core::Result`/`bail!` shim; allocation and error formatting from Candle are not reproduced. This has no effect on successful arithmetic, but nonfinite timings are diagnostic values only. The candidate uses unsigned integer maximum of sign-cleared F32 bits, checks the final maximum against infinity, then runs the original quantization code. Scalar tails also participate in the maximum before the check.

```sh
python3 scripts/benchmark_laya_peak.py --directory artifacts/runtime-comparison/new-peak-build
python3 scripts/measure_laya_peak_ic.py --build artifacts/runtime-comparison/new-peak-build --directory artifacts/runtime-comparison/new-peak-proof
```

Run from IC-Imajev. Output paths must be new. Build needs Cargo, wasm32-unknown-unknown and Node; no external Rust dependencies are needed. `--laya-root` selects another Laya checkout. The source and license are read without changing that checkout. The generated control/candidate, source archive and module hashes are retained in the build directory.

Node validates 128 cases against both copies and an independent scalar reference. Cases include half-way values and adjacent F32 values, zero, signed zero, subnormals, maximum finite values, Inf, quiet/signaling NaN, negative NaN and scalar tails. Nonfinite values are at the final position, including the final row. Early nonfinite rejection has different scan work and is not a performance claim.

The IC runner checks for a loopback local network, creates one detached diagnostic canister, and runs 36 cases with alternating backends three times: 216 ordinary queries plus 36 preparation updates. It compares output/scale digests and rejection status to the Node proof, checks repeated instruction counts and stops/deletes only the created canister. If abruptly killed, use the persisted canister ID for manual cleanup. Diagnostic exports have no production authorization or persistence contract.

The counter includes per-run output allocation, row quantization and diagnostic byte assembly. Preparation, response hashing and replies are outside the counter. These counts measure one same-module A/B diagnostic, not full Laya inference. The ready-to-apply source patch is [int8-unsigned-peak.patch](../../patches/laya/int8-unsigned-peak.patch); validate complete-model logits and instruction counts before adopting it.
