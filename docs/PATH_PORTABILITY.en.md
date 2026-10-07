# Local path portability

[日本語](PATH_PORTABILITY.md)

On October 7, 2026, machine-specific external-drive and home-directory paths were removed from shared documents, provenance records, and execution scripts. The repository root is derived from `__file__`, and decision code is imported from the local `tools/proposal_assessment/` package. These changes affect environment configuration; numerical compression, input budgets, and decision gates remain unchanged.

## Commands requiring external data

Data specified by these arguments is read-only input. Missing required arguments stop execution before experiment outputs or canister operations begin.

|Script|Argument|Input|
|---|---|---|
|`analyze_laya_cost.py`|`--laya-root`|Laya checkout for comparison|
|`summarize_optimizations.py`|`--laya-root`|Laya checkout containing reference sources|
|`benchmark_laya_kernels.py` / `benchmark_laya_peak.py`|`--laya-root`|Pinned Laya sources and weights; also specify a fresh output with `--directory`|
|`make_benchmark.py`|`--source`|Retained `boomdao_query_benchmark/summary.json`|
|`report_proposal_assessment.py`|`--source`|Retained `boomdao-600-660/v1` directory|
|`benchmark_proposal_assessment.py`|`--source-root`|Root containing the historical benchmark archive; defaults to this checkout|
|`checkpoint_goal_recovery_baseline.py`|`--backup`|New directory for a snapshot download; its parent must already exist|

For example, with a read-only Laya checkout beside this repository:

```sh
python3 scripts/analyze_laya_cost.py --laya-root ../IC-Laya-Standalone
python3 scripts/make_benchmark.py --source ../IC-Laya-Standalone/artifacts/boomdao_query_benchmark/summary.json
```

The recovery snapshot destination is never chosen automatically. These examples explain argument usage; snapshot operations and inference were not performed during this portability work.

`benchmark_proposal_assessment.py` reads `snapshots/proposal-ID.json` inside the archive selected by `--source-root`, even when the manifest retains historical absolute paths. A missing relocated file stops execution; the script does not fall back to the original absolute path. It rejects references outside the archive and verifies the content SHA-256, proposal ID, and SNS root. The frozen manifest remains unchanged.

## Original and current sources

`tools/proposal_assessment/UPSTREAM.json` identifies upstream files by repository name, relative path, and original hash. `local_sha256` identifies the current local copy. The `adapted` field distinguishes changed CLI defaults from unchanged decision code.

The 33 pre-change files were retained locally in `artifacts/path-portability-20261007/frozen/`, with hashes and archive locations in `original-sources.json`. Historical frozen manifests and source hashes were not rewritten to present current sources as those used by earlier experiments. If an old driver's source checks reject current files, verification requires the retained historical sources. New experiments must use fresh outputs and hashes of current sources.

The shared JSON for historical GPT reference results normalizes only command executable and output paths. Predictions, token usage, and measurements remain unchanged. `path_export` records the original hash and transformation scope; the original bytes are included in the local frozen archive.

## Retained temporary paths

Six temporary-path occurrences were retained, separately from machine-specific home and drive paths:

- Four occurrences in `test_adaptive_graph.py`, `test_packed_graph.py`, and `test_limit_fallback.py` are dummy paths used when building fallback commands. They are not actual output destinations.
- One occurrence in `test_review_fixes.py` is a dummy path for frame-capacity checks with a mock transport. It writes no files.
- One occurrence in `finalize_proposal_assessment.py` checks the deletion boundary when restoring historical SSD staging. It matches existing historical metadata. Substituting a generic temporary directory would broaden the cleanup boundary, so this guard remains. The restoration procedure was not executed during this work.

## Verification

Required sources were copied into a temporary checkout with a different name and a space in its path. Checks covered ROOT resolution, local imports, CLI defaults, required external arguments, and unchanged benchmark inputs. No Git worktree was created. Retained decisions and frozen metadata were also checked; no inference or canister operations were performed.

Before/after source hashes and verification results are in the [portable verification record](evidence/path-portability-20261007/verification.json). This shared record excludes weights, private keys, and original files containing historical machine-specific absolute paths.
