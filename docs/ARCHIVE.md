# Historical experiment archive

The OSS cleanup separates **1,387 script/source files and 175 documents** from the public working tree. Selected current build/input/test entry points and their dependencies remain. This is a scope decision, not proof that every archived experiment is obsolete.

## Archive identity

- Source revision: `4eb568f90683b5b3e60fdaefb914a4e37c9d1ec4`.
- Archive: `experiments-and-docs.tar.gz`.
- SHA-256: `4bf80a681ad58651d4d572a668e9b8d8028e93ea330a66999d0dd9cdab9f0bc8`.
- Original tracked files preserved: 1,653.
- The archive includes original `scripts/`, `docs/` and entry-point READMEs before translation, including files also retained in the public tree.
- [experiment-archive.json](experiment-archive.json) records the archive identity, file counts and digest of the original detailed inventory. The detailed path/hash lists are kept in the private local archive.

The archive has been created and byte-verified locally under `.local-only/oss-archive/4eb568f90683/`. It is ignored by Git and has **not been uploaded or published**. Distribute it as a separate reviewed research archive if needed; do not add it to the minimal source tree. It contains historical material that still needs review before separate publication.

## Reading and restoration

The publication cleanup on October 13, 2026 also separates the historical GPT comparison run and old frontend inspection notes. Their original bytes and the detailed experiment inventory were copied and SHA-256 verified in an ignored local archive before removal from the public tree. The comparison tools accept an explicitly supplied reference JSON; those historical results are not ordinary build inputs. See the [assessment guide](../tools/proposal_assessment/README.md).

Obtain the archive, verify its SHA-256, and extract it into a separate directory. Paths inside it are relative to the original repository root. Preserve that layout if inspecting historical drivers: many derive the root from their file location. Do not extract it over the current public checkout, because that would overwrite English documentation and restore excluded files.

The bundle preserves tracked source and documentation. It does not include weights, keys, original inference outputs, frozen binaries, snapshots or other ignored artifacts. Historical drivers may require exact frozen source hashes and module identities, and some perform canister updates or staging cleanup. Extraction is not an instruction to run them.

## Selection and verification

Selection follows executable references and selected CLI/test entry points, resolving explicit repository/relative paths and Python imports. It preserves actual generator input sources rather than every unrelated `lib.rs`, `Cargo.toml` or benchmark file with the same basename. Three dynamically imported window preparers remain because regression tests exercise them. A second pass deleted 170 additional obsolete benchmark runners, standalone crate manifests/locks and old measurement scripts. The public tree retains 63 original script/source files plus the publication checker, its tests and a usage guide.

One historical directory name, `scripts/lane_pair_bench/`, remains only for the kernel input read by the maintained SIMD generators. It no longer contains benchmark executables or a Cargo project.

The archive separation preserved retained runtime source and evidence JSON. Later CLI clarification adds historical guards and named entry points; the original archive still preserves pre-change source bytes. Historical Japanese notes, progress logs, one-off kernel experiments, old proof/report drivers and SSD staging restoration utilities are in the archive. The English evaluation report links to this guide for archived supporting notes. Original provenance hashes remain associated with their original bytes.

Removing files from the working tree does **not** remove them from Git history. See the [publication audit](OSS_PUBLICATION_AUDIT.md) before publishing a repository with its existing history.
