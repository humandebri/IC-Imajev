# inference-core

Model-independent primitives extracted from the Imajev runtime and used by `imajev-runtime`. The crate targets native execution and Wasm SIMD128 and has no external dependencies.

| Module | Contract |
| --- | --- |
| `linear` | Token-major inputs and output-major weights; independent tokens share SIMD lanes while each dot preserves column order and multiply-then-add behavior |
| `bf16` | BF16 detection, packing and expansion; `pack` stores the upper 16 bits, so lossless callers must first check `all_bf16` |
| `block256` | Lossless BF16/F32 payloads in 256-element blocks; validates finiteness and canonical encoding and preserves signed zero |

`MAX_FLOATS=900000` and the block codec's 2 MB frame budget preserve implementation limits, not generic model dimensions. Matrix operations validate shapes but do not scan every input for finiteness; callers handle input validation. Discard partial output when `block256::append` fails.

```sh
cargo test --locked -p inference-core
cargo run --locked -p inference-core --example linear_codec
```

See [linear_codec.rs](examples/linear_codec.rs) for usage and [runtime architecture](../../docs/runtime/ARCHITECTURE.md) for boundaries. GGUF loading, tokenization, text generation, model graphs and scheduling are outside this crate. Generalization beyond this model has not been validated.
