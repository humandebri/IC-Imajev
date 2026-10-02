The text runtime primitives in this repository are written for this experiment.
The authoritative model/prompt/readout/calibration behavior comes from
mohit67890/imajev commit a0134749e0900189c129cd6bb5000969f3b64bb5 (Apache-2.0).
Its source is downloaded into ignored vendor/ and source hashes are in MODEL_LOCK.json.
Qwen/Qwen3.5-4B and mohit67890/imajev-4b declare Apache-2.0 licenses.

benchmarks/laya_baseline.json reproduces existing experimental results from
IC-Laya-Standalone (MIT); its license is retained in docs/licenses/Laya-MIT.txt.
Laya tiling, fused execution, client-held splitting and lossless transport designs
were studied as references; the Imajev F32 implementation is written separately.
The loop-unrolled integer row/column macro layout in int8_kernel.rs follows
Laya's MIT-licensed kernel; pointer setup, block256 scales and fusion are adapted
for this experiment. The MIT notice is retained in docs/licenses/Laya-MIT.txt.
No Laya identity, network or canister is reused.
Python and Rust dependency versions are in requirements.lock and Cargo.lock;
these projects retain their respective licenses.
