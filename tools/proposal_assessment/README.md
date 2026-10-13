# Imajev proposal assessment

These tools validate typed SNS snapshots, extract prior values from historical renderings, select relevant fields, hold decisions with insufficient evidence, and compact inputs while retaining changes. Runtime execution uses local assessment code rather than an external Laya checkout or classifier.

The implementation was replaced on October 8, 2026. Retained prompt and field metadata live in `prompt_contract.json`; [PROVENANCE.json](PROVENANCE.json) records attribution. Keep the [MIT notice](../../docs/licenses/Laya-MIT.txt).

Use the explicitly named `scripts/reproduce_proposal_assessment.py --historical` for historical reproduction. Its compatibility driver, `scripts/run_fresh_proposal_assessment.py`, retains module, model, input-hash, layer and query-count verification, but targets a historical frozen module and requires a separately supplied snapshot/run archive. It is not a general driver for the currently deployed canister. Older 600–660 benchmark entry points are compatibility utilities for retained archives.

The experiment checks numeric voting-participation conditions. It does not establish overall proposal safety or correct adoption decisions. Inputs above its 128-token budget become hold rather than being truncated. The 0.6 threshold is uncalibrated; exclusions are not approvals. These tools do not execute votes.

```sh
python3 -B -m tools.proposal_assessment.verify_local_copy --sources-only
PYTHONPATH=scripts:. .venv/bin/python -B -m unittest \
  scripts.test_proposal_contracts scripts.test_proposal_snapshot_archives
```

The GPT comparison JSON is supplied separately from the historical archive; it is no longer distributed with public sources. It remains a benchmark input and does not invoke an API. Preparing the old 600–660 benchmark requires an explicit reference file:

```sh
python3 -m tools.proposal_assessment.benchmark_600_660_binary prepare \
  --output <new-output-directory> --snapshot-archive <snapshot-archive> \
  --reference-json <historical-vote-results.json>
```

The reference file must describe all 61 proposals and match every selected snapshot hash. Historical source hashes and frozen manifests retain their original meaning, including references to files now held separately. See the [script index](../../scripts/SCRIPT_INDEX.md) for CLI gates and side effects, and the [archive guide](../../docs/ARCHIVE.md) for the separated experiment collection.
