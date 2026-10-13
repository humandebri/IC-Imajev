# OSS publication audit

Assessment date: October 12, 2026. Baseline: `4eb568f90683b5b3e60fdaefb914a4e37c9d1ec4`. This is a publication-hygiene review, not a full application vulnerability audit.

## Current status: October 13, 2026

The public source snapshot is published separately at [humandebri/IC-Imajev-public](https://github.com/humandebri/IC-Imajev-public), with a new initial history. The original development repository is retained privately after the authorized cutover; earlier public copies may still exist. The audit below describes that original development repository before cutover, rather than the new snapshot's reachable history. See the [cutover record](GIT_HISTORY_CLEANUP.md).

Before cutover, the repository was already public. References below to keeping development history private describe the intended original publication route. The follow-up audit at main `95062a6` checked 431 tracked working files, 2,851 main-history blobs, 2,889 blobs including 16 fetchable PR heads, and 148 commit messages. No recognized credentials were detected; email-like working-file matches were dummy fixtures and additional historical matches were NumPy matrix expressions. Original history still contains 43 personal-path occurrences on main and one additional occurrence reachable through PR #12, across 14 filenames and 38 blob versions. Four commit email strings remain, three noreply and one ordinary attribution address.

The follow-up removes the stale Bun lock and old frontend inspection notes, separates the detailed GPT comparison log into a private local archive, and replaces the full experiment path list with counts and provenance hashes. Historical benchmarking now takes an explicit `--reference-json`; preserved evidence hashes continue to identify their original bytes. Public neuron snapshots and pinned model provenance remain necessary inputs. Compressed WAT contents and PNG metadata were also inspected. A generated 503-file package of the pre-cleanup tree had no credential or personal-path matches; it contained only the expected sources and selected browser assets.

The source cleanup does not rewrite original history. The [history cleanup record](GIT_HISTORY_CLEANUP.md) describes the separate new snapshot and private retention of that history. The dated sections below retain earlier measurements and should not be read as current file counts or visibility.

## Findings

| Area | Result | Publication consequence |
| --- | --- | --- |
| Current publishable files | No matches for personal home/volume paths, common private-key headers, recognized provider tokens, credential URLs or literal credential assignments | No detected exposure in the checked scope; pattern checks cannot certify absence of secrets |
| Reachable Git blobs | 46 personal absolute-path occurrences across 40 blob versions and 15 original filenames | Existing history still exposes machine-specific locations even after cleanup |
| Deleted operational notes | Earlier `PLAN.md` and `docs/UPDATE_INSTRUCTION_TARGET.md` contain much longer planning/operation histories | Public package excludes Git history; keep existing development refs private |
| Commit metadata | Four distinct author/committer email strings in the locally reachable history | Existing email attribution stays in private development history; the public source package contains no commit metadata |
| Scripts and documents | 1,387 script/source files and 175 documents separated into a verified local archive | Smaller public tree, with original bytes recoverable; archive publication is a separate decision |
| Reproducibility | History-free source package includes exactly verified browser assets; Rust builds directly from source | Browser and optimized candidate builds no longer require private archives; migration still requires full numerical/performance gates |
| Local exclusions | Historical comparison sources were excluded only by local Git rules | Equivalent rules now live in `.gitignore`, alongside environment files, keys and managed worktrees |

Historical path locations include old README/plan versions, local token/chunk reports, instruction-target notes, SHA-256 generation and earlier benchmark/source scripts. The checker reports original filenames, blob IDs and line numbers without printing the matched private values.

Public canister IDs, gateway URLs, deployed domain names, model revisions, module hashes and public governance account identifiers are expected configuration or evidence. They were not treated as authentication secrets. Saved governance data and model input fixtures retain their original content; translating those would change benchmark inputs.

## Changes made

Public entry-point documentation is now English: root/frontend/core/tool/caller guides, API documentation and runtime architecture. A documentation index, contribution/security reporting guidance, script-use inventory, verified archive manifest and repeatable publication checker were added. The initial cleanup left retained executable source and evidence JSON unchanged. The subsequent entry-point work adds CLI guards and historical flag forwarding without changing runtime arithmetic or evidence JSON. A follow-up removed 170 more unreferenced benchmark/measurement files after distinguishing executable paths from generic filename mentions.

The separate [archive](ARCHIVE.md) is local and ignored; its original files were byte-verified before removal. Do not distribute the entire development directory. Ignored directories contain model data, generated artifacts and potentially local credentials/private notes; they were not reviewed as publishable files.

## Checks

```sh
python3 scripts/check_publication.py
python3 scripts/check_publication.py --history
PYTHONPATH=scripts python3 -B -m unittest scripts.test_publication
cargo test --workspace --locked
```

The checker includes tracked working files and untracked non-ignored files, omitting deleted paths. `--history` scans all blobs reachable from local Git refs. It returns a nonzero status for candidates, including the remaining historical paths. The current-tree check passes; history failure is expected and must not be described as a clean history.

Validation completed: standard Rust workspace tests (136 tests) also pass in a 319-file source snapshot containing only publishable working files, excluding all ignored local artifacts; standard Wasm canister build; 51 retained Python regression tests plus three publication-checker tests exercising redaction, credential candidates, ignored files, deleted files, symlinks and Git history; frontend TypeScript check; and six-method public Candid comparison. Markdown links and archive integrity were checked separately. Four retained code generators reproduced the runtime sources byte-for-byte in the public-only snapshot. The follow-up reduced original script/source files from 1,450 to 63, deleting unused benchmark crate scaffolding and runners rather than retaining generic filename matches.

The pre-existing all-feature assertion failure is fixed: the test incorrectly rejected a 90-token stream suffix even though the prefix5 contract accepts suffixes through 91. It now rejects 92 and explicitly tests 1/89/90/91 acceptance, including disabled-feature rejection. Runtime validation was unchanged. All-feature workspace tests pass: 224 passed and one ignored. Default-feature tests also pass from the extracted history-free source package; a standard release-mode Wasm build passes there.

The extracted source package contains no `.git`, model checkpoints, internal artifacts, `.venv` or private archive. Using existing installed Node dependencies, its complete frontend prebuild checks, TypeScript and Vite build pass without those inputs. A fresh `npm ci` download was not performed. The browser assets' complete digest inventory was verified, and changed/missing asset checks reject before build. Standard-library example compaction checks avoid importing the upstream model package; browser tokenization independently checks actual token counts.

## Publication route

Follow [Public source releases](RELEASING.md). Publish the generated reviewed package or initialize a new public repository from its extracted contents; keep the development history and research archive private. No refs were rewritten and no remote was changed or uploaded. Publishing the original repository's existing refs would still expose the historical findings above.

The source release supports standard Rust, pinned browser application and source-only optimized candidate builds. The frozen donor/compiler chain is retained only behind historical opt-in. The public builder compiles the 34 approved WAT kernels with a Cargo-built patcher and no parent Wasm/compiler archive. Model weights and the historical research archive remain excluded. Building a candidate does not approve migration; see the [optimized build verification record](OPTIMIZED_BUILD.md).

## Coverage limits

Initial inventory: 1,871 tracked paths, including 1,450 under `scripts/` and 197 under `docs/`. The first history inventory covered 2,880 reachable blob objects and 147 commits. After pruning, the current-tree check covers 319 publishable files with zero candidates. The latest history run covered 2,891 reachable blobs with the same 46 historical path matches. No recognized secret-token/key-header matches were found in the scanned working tree or reachable blobs.

Detection is format-based and does not include entropy-based secret detection, semantic review of every string, compressed payload contents, ignored local files, reflogs, dangling objects, unfetched remote refs, other repositories or hosting-provider caches. Commit email metadata was counted separately. A clean result is not a guarantee of safe publication.

## Script entry-point follow-up

The [complete script index](../scripts/SCRIPT_INDEX.md) records each retained source's role, target and effects. Preferred command names are `build_optimized_inference.py`, `reproduce_proposal_assessment.py` and `reproduce_prefix_inference.py`. Historical CLIs and the legacy optimized build branch require `--historical` before optional imports, artifact operations or canister calls; historical help without opt-in is available through standard Python alone. Imported helper APIs remain unchanged.

Known retained historical child/fallback CLI calls forward opt-in explicitly; unrelated frozen executables keep their original arguments. Source-hash checks and original frozen evidence are not bypassed. Guard helper identity is included in new assessment source-hash captures. Validation: 60 Python tests pass, including blocked execution, help without model dependencies, aliases, flag consumption, child forwarding and fallback propagation. The final working-file check covers 329 publishable files with zero candidates; this check includes the named entry points, usage guard/tests and index. No historical experiment, owner update, live inference or deployment was executed during this work.

## Public release follow-up

`package_public_release.py` selects current publishable sources plus only the 70 allow-listed checksum-verified browser assets and the pinned tokenizer license. It rejects unsafe paths, symlinks and publication-pattern candidates, uses fixed archive metadata, includes a file digest inventory and refuses overwriting an existing output. Bundle safety and example-content regression tests supplement the retained tests. Ordinary frontend builds verify supplied assets instead of loading frozen prefix/model inputs. The [release guide](RELEASING.md) describes extraction, clean-history publication, frontend/Rust builds and the optimized-module reproduction limit.

Final validation: 84 retained/publication Python tests pass. The current-tree scan covers 333 publishable files with zero candidates. The public tarball contains 405 files, including its inventory, 70 pinned browser assets and the upstream license. No upload or deployment was performed.
