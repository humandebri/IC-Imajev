# Historical reproduction inputs

Current source builds and browser releases do not require historical experiment projects. Historical CLI entry points require `--historical` and separately supplied inputs; see the [script index](../scripts/SCRIPT_INDEX.md) and [assessment guide](../tools/proposal_assessment/README.md).

## Input requirements

Historical investigations may require an original source snapshot, exact frozen module/compiler inputs, model data or recorded measurement results. Obtain the required inputs separately and verify their original hashes and model/module identities. Preserve relative layouts and read each driver's help before execution. Do not extract old sources over a current checkout or edit frozen manifests to accept changed bytes.

The GPT comparison benchmark takes an explicit `--reference-json` and checks it against the selected proposal snapshots. The public tree does not include detailed comparison run logs or historical project inventories.

## Public evidence

The [evaluation report](PROPOSAL_FILTER_REPORT_20261007.en.md) and [evidence index](evidence/proposal-filter-20261007/README.md) retain the numerical scope and provenance needed to interpret historical measurements. Original/export hashes identify their original bytes, including references to inputs no longer distributed here. These measurements do not identify or validate a newly deployed module.

Keep model weights, credentials, local operation records and frozen binaries outside the public source tree. Source packages include only current reviewed sources, approved kernel inputs and checksum-verified browser assets. See [Public source releases](RELEASING.md).
