# Published Git history cleanup

The October 13, 2026 cutover uses [humandebri/IC-Imajev-public](https://github.com/humandebri/IC-Imajev-public) for a new public source history and retains the original `humandebri/IC-Imajev` repository privately. The original repository was public before this cutover. Its main history contains 43 personal absolute-path occurrences; PR #12 exposes one additional occurrence. Existing commit email attribution remains in that private original history. No recognized authentication secret was detected in the checked scope. Earlier public copies cannot be revoked.

## Prepare and verify

1. Finish and review the source cleanup on its own branch. Run publication checks and the affected regression tests.
2. Preserve the original repository in a private offline Git bundle, including needed branches and tags. Separately preserve uncommitted work, managed worktrees and ignored local archives; a Git bundle does not include those files. Verify the backup before any remote change.
3. Generate the deterministic source package with `scripts/package_public_release.py --output <new-ignored-output>`. Inspect its manifest and verify every extracted file against it. This package excludes Git history and arbitrary ignored data while deliberately including the pinned browser assets.
4. Initialize the replacement repository from that reviewed package or its reviewed source subset. Do not copy `.git`, existing refs, the offline bundle or local archive. Choose the initial commit's attribution explicitly. The package contains generated browser assets; decide whether to distribute them in Git or as a matching release package and keep the documented build route consistent.
5. Check the new repository's complete reachable history, tree contents and release manifest. Verify the selected build and CI route from an empty checkout/extraction. Record the source package hash, initial commit and final release identity together.

## Selected remote cutover

The maintainer authorized creating a new clean public repository and making the old repository private after reviewing the cleanup and source package. The new repository starts with a snapshot commit using GitHub noreply attribution; it does not import development refs or PR history. The matching source bundle is distributed as a release asset, so browser assets remain outside normal Git tracking. Build CI is copied with the source. No deployment workflow or deployment credential is transferred. Future changes and reviews should target the new repository. The old working checkout and private backup remain available for historical investigations.

An alternative is replacing refs in the current repository. Force-pushing main alone does not remove old PR refs, cached views or external clones. Do not mark that alternative complete merely because `git log main` has one commit. Hosting-provider cleanup must be assessed separately, and existing copies cannot be revoked. Confirm the exact refs and force-push action before using this approach.

Verify the new repository's visibility, reachable history, source contents, release asset hash and CI after cutover. Original refs are preserved privately without force-pushing them. Canister upgrades and site deployments are outside this operation.
