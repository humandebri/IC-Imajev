# Design and validation entry points

IC-Imajev executes a pinned text decision model on Internet Computer. Changes preserve model identity, tokenizer configuration, quantization rules and accumulation order, and are checked against independent references and actual Wasm where applicable.

The canister manages model upload, caches, authorization, paid inference and receipts. The runtime implements INT8 projections, F32 LoRA, Attention, DeltaNet and MLP. Clients prepare inputs, execute queries and validate results.

Use the [documentation index](docs/README.md), [runtime architecture](docs/runtime/ARCHITECTURE.md), [API guide](docs/CANISTER_API_NAMES.md), [model lock](MODEL_LOCK.json) and [license notice](NOTICE.md). Historical optimization plans are in the [separate experiment archive](docs/ARCHIVE.md).

Measurements must identify inputs, module, reference and accounting scope. Local instructions and timings do not establish production subnet performance, and kernel speedups do not establish whole-model speedups.
