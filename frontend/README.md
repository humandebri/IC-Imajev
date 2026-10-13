# IC-Imajev decision playground

The React frontend prepares text inputs in a Web Worker and executes anonymous inference queries against the canister pinned in `src/inference-release.json`. Cloudflare serves static assets. The public demo is https://imajev.kinic.xyz; this guide does not verify its current live state.

## Prerequisites

Use Node.js 24 or newer, npm and standard Python 3. Extract the matching [public source release](../docs/RELEASING.md), which includes checksum-verified tokenizer/readout and prefix assets. A plain Git clone needs these two generated directories from the matching package: `public/tokenizer/` and `public/inference/`. Ordinary browser builds do not need model weights, NumPy or `.venv`.

`package-lock.json` is the supported dependency lock; use `npm ci`. Alternate package-manager locks are not maintained.

For explicit maintainer regeneration only, prepare the tokenizer/readout files listed in `MODEL_LOCK.json`, use `.venv/bin/python` with NumPy, and supply `checkpoints/full-int8.manifest.json` with a matching five-token prefix archive at `artifacts/browser-prefix5-v1/`. Then run `npm run prepare:tokenizer` and `npm run prepare:inference`. The exporter checks the module, cache and generated manifest against the pinned release; those internal inputs are not included in a clean clone.

## Development and checks

```sh
cd frontend
npm ci
npm run typecheck
npm run dev
```

Open http://127.0.0.1:5173. The `dev`, `build` and `test` lifecycle hooks verify the supplied pinned browser assets, generate public Candid, verify saved example compaction/token counts, and check query profiles. Generated `public/tokenizer/`, `public/inference/` and `public/api/` are ignored by Git.

```sh
npm run build
npm run preview
npm test
npm run test:query
npm run test:candid
npm run test:examples
```

UI tests require Playwright Chromium. Query fixture tests additionally require retained local validation evidence. Commands named `test:live-query`, `test:boom-live` and `test:boom617-live` send explicit anonymous queries to the configured public canister; they are separate from offline source checks. No deployment is implied by a test or build.

## Inputs and execution

A question and 2–7 choices are required; evidence is optional. Empty or duplicate choices, the reserved `__unknown__` label and choices longer than 128 UTF-8 bytes are rejected. The five-token prefix plus a 1–91-token suffix gives a total budget of 6–96 tokens, including formatting and choices.

The runner selects a measured query plan pinned to the module, performs readiness and module checks, and executes dependent carries in order. Query counts come from `src/query-profiles.json`; the documented release uses 32–63 inference queries. Results show the selected choice, probabilities, unknown probability and abstention. Editing the form does not change an active run's frozen input.

Prefix assets use content-addressed URLs and verified hashes. Worker caches are reused within a page and prefetching is cancelled per run. Each execution has its own carry and cancellation state. `Cancel` stops subsequent requests and discards stale replies; it cannot guarantee interruption of a request already executing on the canister. A query times out after 30 seconds and a full run after five minutes, without automatic inference retries. Input history is not persisted by the UI.

The agent verifies node signatures and checks module/model/pack and prefix identities. Host-checksum responses are accepted only after signature and header validation. Model weights and owner credentials are not shipped to the browser.

## BOOM examples

Saved public source data in `data/` defines the saved short English examples checked at build time. Proposal #584 demonstrates a metadata title whose payload transfers treasury funds; its rejected status does not imply that the transfer executed. #653 applies an additional-mint scenario to saved supply and account values. #617 compares voting eligibility using saved indexed neuron data. Neurons are not people, snapshots are not atomic, and these examples do not establish intent or general attack-detection accuracy.

```sh
npm run refresh:boom-ledger
npm run refresh:boom-neurons
npm run test:examples
```

The refresh commands fetch public data and update local fixtures. Ordinary builds use saved data and do not refresh it from the network. Historical measurements in the separate archive apply to their recorded modules and inputs.

## Deployment

`cloudflare.config.ts` contains the project's public worker name and custom domain. Forks should configure their own worker/domain. Use the `cf` CLI and inspect available authenticated profiles before selecting one; an unauthenticated default does not mean all profiles are unauthenticated.

```sh
npm run build
node scripts/prepare-cf-output.mjs
cf --profile <authenticated-profile> deploy --prebuilt
```

`npm run deploy` uses the default profile. Deployment changes a remote site and is an explicit operation. Coordinate any canister upgrade, heap-cache preparation, prefix assets and frontend release pins. See the [public query API](../docs/PUBLIC_QUERY_API.md).
