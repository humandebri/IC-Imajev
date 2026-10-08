# Imajev mainnet deployment — 2026-10-07

## Target and scope

- Network: `ic`; canister: `xis3j-paaaa-aaaai-axumq-cai`.
- Controller / initialized runtime owner: `r75h6-lqd7b-5jack-at55d-vvti2-lg5qy-ly73a-5ezve-odnkc-kagu3-nae` (`llm-wiki-mainnet`).
- The human explicitly authorized replacing the prior Hyperliquid Testnet Vault/Core/Policy/Journal application by **reinstall without creating a snapshot**. That reinstall succeeded.
- Model upload, cache preparation, and actual inference validation were subsequently authorized with **a fee-inclusive ceiling of 6.5 ICP for this model phase**. This amount has already been converted and deposited; no further automatic minting or top-up is allowed.
- Controller changes, additional canisters, public paid-service activation, and frontend API integration are not authorized by this deployment phase.

## Pinned artifacts

| Item | Location | SHA-256 |
|---|---|---|
| Frozen measured runtime | [Original Wasm](../artifacts/paid-update-v1/build-v3/full.wasm) | `c80e77130301636411b6df230f7ce04892898c5ba694e242b4c390c6a8d746ff` |
| Installed lossless compression | [Gzip Wasm](../artifacts/mainnet-reinstall-20261007/full.wasm.gz) | `5e2a0b101d4dde0ff503a8c7d399efc17e7db7618dc2c3aa6eddee49a771b3a9` |
| Text INT8 pack | [Model pack](../checkpoints/full-int8.pack) | `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84` |
| Manifest | [Manifest](../checkpoints/full-int8.manifest.json) | `f8403edc1de2a9268ee79f8cb6b7b37d9dba0ebb66ee8655dd934ebe4be4f431` |

The compressed and original runtime were verified byte-identical after gzip decompression. The IC installed module hash is the hash of the submitted gzip bytes. Drivers must compare against `5e2a…`, not the raw Wasm hash. Ordinary Cargo builds from the existing local configuration do **not** reproduce this frozen optimized runtime.

The 4,702,451,200-byte pack already includes the original F32 LoRA and trained readout. It uses 2,613 upload chunks of at most 1,800,000 bytes. Uploads and whole-pack hashing are owner updates; the runtime sets `ready` only after its SHA-256 matches the manifest. No further reinstall or upgrade should occur during an unfinished upload.

## Resource authorization and empirical cost correction

The target is reported as a 13-node application subnet by the [Dashboard canister API](https://ic-api.internetcomputer.org/api/v3/canisters/xis3j-paaaa-aaaai-axumq-cai) and [subnet API](https://ic-api.internetcomputer.org/api/v3/subnets/brlsh-zidhj-3yy3e-6vqbz-7xnih-xeq2l-as5oc-g32c4-i5pdn-2wwof-oae).

Wasm memory limit was explicitly authorized and changed from 3 GiB to 4 GiB. Compute and fixed memory allocations remain zero; controllers are unchanged.

**Do not reuse the initial USD4/month estimate.** The pilot's actual `idle_cycles_burned_per_day` growth gives approximately **317,500 cycles/GiB/second** for this target, rather than the earlier official-table assumption of 127,000. The guard uses a conservative 320,000 for projected growth and the live observed daily rate for existing allocation.

The 32-chunk / 57.6 MB pilot consumed 124,636,706,085 cycles including manifest allocation/preparation and observation overhead. Extrapolated total upload cost is about 10.18T cycles, not a guaranteed invoice. Hash verification, cache preparation, and inference are additional. Based on the previously measured prepared inference heap, projected idle maintenance is about **6.9T cycles per 30 days** (approximately USD9.4 at the illustrative USD1.37/T conversion).

After this measurement, the human explicitly selected **freezing threshold 30 → 14 days**, with no extra funding. This changes the minimum protected reserve, **not the storage price or monthly maintenance cost**. The live setting is 1,209,600 seconds. See the [pilot projection](../artifacts/mainnet-deploy-20261007/pilot-cost-projection.json) and [authorization record](../artifacts/mainnet-deploy-20261007/authorization-14days.json).

## CLI-only drivers

- [Offline codec](../crates/imajev-client/examples/mainnet_candid.rs): Candid encode/decode only; no network or keys. Requests were cross-checked with `didc`.
- [CLI transport](../scripts/mainnet_cli.py): explicit target, identity, network, pinned module/controller checks, query receipts, bounded safe retries, and cycle guards. It never exports private keys.
- [One-shot funding](../scripts/mainnet_fund.py): fixed 6.4999 ICP mint plus 0.0001 ICP ledger fee; persistent intents prevent blindly repeating financial operations after unknown outcomes. **Already executed; do not run to mint again.**
- [Uploader](../scripts/mainnet_upload.py): resumable chunks and whole-pack SHA validation. A query may reach a lagging replica after certified updates, so known committed progress is confirmed before costly replay.
- [Bootstrap / owner inference](../scripts/mainnet_prepare_inference.py): 721 immutable weight updates, 24 exact common-prefix dense states, and 64 prefix-bank registrations. Voting38 is registered first, common27 last. The experimental owner API receives **suffix token IDs only**, unlike the paid API's full token input.

`prepare_fixed_prefix_state` exists in the frozen module but is missing from the older hand-maintained paid DID. The preparation driver generates a dedicated exact DID for this owner method. The remaining calls use the existing paid DID.

Prefix registration and owner inference start/continue are not safely replayable after an unknown outcome. The driver halts rather than duplicating them. A successful cached response can be resumed from the archived receipt. There is no owner inference progress query for repairing arbitrary lost state.

## Readiness and validation boundaries

A running Wasm module alone is **not** model readiness. The pack, all caches, and prefix banks must be prepared before inference. Paid `enabled` remains false throughout this owner-validation phase.

The three basic UI inputs have 74/72/72 total tokens and suffix lengths 47/45/45. Their cached 27-token prefix was verified against the actual saved prefix embedding request. Native BF16/F32 measurements are input provenance only; they are not imposed as expected IC results. Actual mainnet outputs must be separately measured and recorded.

The Cloudflare UI remains disconnected from inference until a separate integration is implemented. Browser ingress cannot directly attach cycles to the paid API. Existing owner-only APIs must not be exposed publicly by leaking the controller key.

## Known temporary storage

Failed raw-Wasm installation attempts left about 3 MiB of management upload chunks. The current `icp` v1.0.0 direct gzip install path does not clear that store, and generic direct management calls lack effective-canister-ID routing. Those chunks are not model weights or a snapshot; their storage overhead is included in live measurements. Do not add a controller proxy, export a key, or change tools solely to clear them without explicit authorization.

## Evidence

- [Runtime reinstall verification](../artifacts/mainnet-reinstall-20261007/verified.json)
- [Funding verification](../artifacts/mainnet-deploy-20261007/funding-verified.json)
- [Pilot measurement](../artifacts/mainnet-deploy-20261007/pilot.json)
- [Latest upload progress](../artifacts/mainnet-deploy-20261007/upload-progress.json)

This document is an operational record, not a claim that unfinished stages are ready or that native and INT8 model outputs are bit-identical.
