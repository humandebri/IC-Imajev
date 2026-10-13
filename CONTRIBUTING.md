# Contributing

Describe the concrete behavior changed and include focused validation. Preserve pinned model identity, numerical contracts, authorization and caller-scoped receipts. Keep measurements tied to their inputs, reference and actual Wasm module.

```sh
cargo test --workspace --locked
python3 scripts/check_publication.py
```

For frontend changes, run `npm ci`, `npm run typecheck` and appropriate tests from `frontend/`. Use checksum-verified browser assets from the matching [public source package](docs/RELEASING.md); maintainers can explicitly regenerate them from retained inputs. Document unavailable validation rather than changing pinned hashes to bypass it.

Do not commit credentials, model weights, generated packs, frozen binaries or private operation records. Use fresh output paths for experiments. Retain attribution and licenses. Source build, canister upgrade, cache preparation and site deployment are separate operations.

Historical drivers and notes are preserved in a [separate archive](docs/ARCHIVE.md). If a change requires one, identify its original path, archived hash and data prerequisites instead of silently restoring an entire experiment collection to the public tree.
