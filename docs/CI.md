# CI

GitHub Actions runs `rust`, `rust-all-features`, `wasm`, `python`, `frontend`,
and `publication` on every pull request, on pushes to `main`, and on manual
dispatch. All six jobs run without path filters and are required on `main`. A newer push
to a PR cancels its previous run. Toolchains and action revisions are pinned.

## Run locally

Use the versions in `rust-toolchain.toml`, `.node-version`, and `.python-version`.
Rustup installs the selected Rust toolchain and `wasm32-unknown-unknown` target.
Install frontend dependencies with `npm ci --prefix frontend`, then run:

```sh
python3 scripts/ci_check.py all
# Or one suite:
python3 scripts/ci_check.py rust
python3 scripts/ci_check.py rust-all-features
python3 scripts/ci_check.py publication
python3 scripts/ci_check.py wasm
python3 scripts/ci_check.py python
python3 scripts/ci_check.py frontend
```

The frontend entrypoint is also available as `npm run test:ci --prefix frontend`.
It deliberately avoids the asset-generating `pretest` and `prebuild` hooks.

| Job | Coverage |
| --- | --- |
| `rust` | Default workspace feature set, unit/integration/doc tests, and the `paid_update_args` example test. |
| `rust-all-features` | Workspace unit/integration/doc tests with all features enabled. This is test coverage, not the production build configuration. |
| `publication` | Current publishable files and complete fetched Git history, including bounded WAT gzip expansion; redaction, unsafe input, archive and deterministic packaging regression tests. |
| `wasm` | Release compilation of the canonical canister for `wasm32-unknown-unknown`. |
| `python` | Explicit standard-library suite: paid transport, public API/build contracts, optimized builders, original paid instruction evidence, and browser example compaction. Runs with site packages disabled. |
| `frontend` | TypeScript, public Candid parity, mocked query scheduling, all 91 published profiles and budgets, calibration approval policy, and rejection of invalid query evidence/prefix assets. |

These checks need only tracked source files and installed dependencies. No model
weights, replica, credentials, `artifacts/`, `checkpoints/`, or `.local-only/`
are required. Dependency/toolchain installation needs network access.

## Retained Python tests

Keep `test_paid_update_transport.py`, `test_public_query_build.py`,
`test_build_paid_contract.py`, and `test_build_prefix_contract.py` with their
retained production tools and optimized builders. Together with API naming and
browser example tests, optimized build/paid regression tests and the replica instruction, intermediate-state and live-contract evidence tests, these form the standard-library CI suite.

`test_paid_prefix5.py` and `test_paid_token_proof_cleanup.py` belong with their
archived proof/measurement scripts and are not required by basic CI. Payment,
receipt replay, refund, and upgrade unit coverage remains in the Rust suite.

## Additional verification

The basic CI does not run model inference, browser E2E, or every individual
feature combination. The all-feature job checks the combined feature set;
production builds retain their explicit feature selection. The recorded carry
test explicitly marked ignored in Rust also needs its external fixtures. Existing asset-dependent commands retain their
full checks:

- `npm run test:query --prefix frontend`: prefix integrity/cache and recorded-wire replay.
- `npm run test:calibration --prefix frontend`: approval policy plus reconstruction
  of the published catalog from the 91 recorded measurements.
- Frontend build and Playwright: require pinned tokenizer/prefix assets.
- Replica proofs: verify payment, refund, replay, and upgrade behavior in a canister.

The Wasm job is a compilation check, not reproduction of the optimized deployed
Wasm. The source-only optimized builder and its separate migration gates are
documented in [Optimized build](OPTIMIZED_BUILD.md); frozen historical builders
still depend on archived inputs.

Additional arithmetic and migration tools do not run in this basic CI. Pinned
browser assets are supplied through the reviewed source package. The `main`
protection requires all six jobs from GitHub Actions and a PR,
with zero required approving reviews for solo maintenance. Force pushes and
branch deletion are disabled, and the restrictions apply to administrators.
Repository Secret scanning and Push protection are enabled separately. These
settings perform no canister upgrade or site deployment.
