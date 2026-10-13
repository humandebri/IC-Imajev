# Public browser query API

The frontend release is pinned in `frontend/src/inference-release.json`, including canister, gateway, module, model, pack and prefix-manifest identities. This guide describes checked-in configuration; it does not verify the current live canister. Public endpoint identifiers and module hashes are configuration, not credentials.

`canisters/inference/public-query.did` is the browser subset: `getModelStatus`, `getWeightCacheStatus`, `runInferenceStep`, `runDeltaInferenceStep`, `runAttentionInferenceStep` and `runFinalInferenceStep`. The full owner and paid service is separate. See the [API method guide](CANISTER_API_NAMES.md).

These methods take protocol blobs, not raw question text. The pinned tokenizer must encode the complete prompt, choices and formatting. The browser contract uses a five-token prefix plus 1–91 suffix tokens, totaling 6–96. Supply 2–7 distinct nonempty choices, excluding `__unknown__`, each at most 128 UTF-8 bytes.

Use `frontend/src/inference-client.ts`: `infer({ state, question, options }, onProgress)` returns a `promise` and `cancel()`. Integration needs Worker bundling, verified tokenizer assets, prepared prefix assets and the matching query profiles. It is not a published standalone npm SDK. Keep carries separate between executions. Preserve signature, module identity, header and asset-hash verification; checksums alone do not authenticate responses.

The runner selects a measured plan from `frontend/src/query-profiles.json`, performs readiness/module checks and executes carries in dependency order. Do not extrapolate tested plans to arbitrary lengths or reuse an unrelated module's measurements. The documented release uses 32–63 inference queries.

## Generate and check Candid

```sh
cd frontend
node --experimental-strip-types scripts/prepare-public-candid.mjs --check
```

The generator compares Candid with the agent's types and exports the public asset during build. Use its explicit write mode only when intentionally updating the canonical definition. Generated browser assets are ignored by Git. The [frontend guide](../frontend/README.md) identifies prerequisite packs/prefix evidence and offline versus live tests.

If embedding `candid:service` metadata in a canister, generate the full service for that exact build; do not embed only the browser subset. Metadata changes alter the module hash. Any deployment must coordinate the canister module, cache preparation and frontend pins.
