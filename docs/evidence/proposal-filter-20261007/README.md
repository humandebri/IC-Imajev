# Historical proposal-filter evidence

These retained exports support the [October 7, 2026 English report](../../PROPOSAL_FILTER_REPORT_20261007.en.md). They are historical records, not a fresh execution attestation or a specification of the current release.

| Record | Contents |
| --- | --- |
| [provenance.json](provenance.json) | Original/export hashes, path normalization and projection scope |
| [decisions.json](decisions.json), [inputs.json](inputs.json), [review.json](review.json) | Decisions, model inputs and conditional old/new evidence reviews |
| [verification.json](verification.json), [optimization-verification.json](optimization-verification.json) | Recorded execution and numerical checks |
| [plan.json](plan.json), [fresh-summary.json](fresh-summary.json) | Prefix/suffix plans and query accounting |
| [build.json](build.json), [execution-policy.json](execution-policy.json) | Historical modules, build inputs and reuse policy |
| [pack-manifest.json](pack-manifest.json) | Quantized pack identity |
| [MODEL_CARD.md](MODEL_CARD.md), [RELEASE-SPEC.md](RELEASE-SPEC.md) | Pinned upstream model descriptions |

Original Japanese reports referenced by `provenance.json` were moved to the [experiment archive](../../ARCHIVE.md). The JSON records remain byte-for-byte unchanged; their hashes still describe their original exports. Weights, original run directories, frozen Wasm and private keys are not included here.
