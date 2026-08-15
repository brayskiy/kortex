# Branching model

```
  work branch ──PR──► develop ──PR──► main
   (feature/…)         (integration)   (production)
```

Two long-lived branches, and all changes flow one way through pull requests.

## Rules (enforced by CI + branch protection)

1. **`main`** is updated **only** by merging **`develop`**. Direct pushes are
   prohibited; a PR into `main` from any branch other than `develop` is rejected.
2. **`develop`** is updated **only** by merging a **work branch** (e.g.
   `feature/…`, `fix/…`). Direct pushes are prohibited; a PR into `develop` from
   `main` is rejected.
3. A PR can be **merged only after the test suite passes** (the `test` check),
   and the `branch-policy` check confirms the merge direction is allowed.

## How this is enforced

- **GitHub branch protection** on `main` and `develop`: requires a pull request,
  requires the `test` and `branch-policy` status checks to pass, and blocks
  direct pushes (including for admins).
- **`.github/workflows/ci.yml`** runs the JUnit suite on PRs to both branches
  (the `test` check).
- **`.github/workflows/branch-policy.yml`** fails the PR if the source→target
  direction isn't allowed (the `branch-policy` check). Branch protection can't
  restrict a PR's source branch, so this workflow does.

## Day-to-day workflow

```bash
# start work off develop
git checkout develop && git pull
git checkout -b feature/my-change

# ...commit...
git push -u origin feature/my-change
gh pr create --base develop --head feature/my-change   # tests + policy run

# after review/tests pass, merge in the GitHub UI or:
gh pr merge --squash --delete-branch

# release: promote develop to main
gh pr create --base main --head develop --title "Release"
gh pr merge --merge        # allowed only from develop, only if tests pass
```

## Releases

When a `develop → main` PR merges, **[release.yml](.github/workflows/release.yml)**
automatically tags a **calendar release** (`YY.WW.patch`, ISO week-date) and
publishes a GitHub Release with auto-generated notes — no manual tagging or
version markers needed.

| When | Tag |
|------|-----|
| first release in an ISO week | `YY.WW.01` (e.g. `26.34.01`) |
| next release that same week  | `YY.WW.02`, `YY.WW.03`, …    |
| first release next week      | `YY.(WW+1).01`             |

`YY`/`WW` come from `date +%g`/`+%V` (so the year rolls over correctly at the week
boundary) and the patch is zero-padded to two digits, starting at `01` each week.
(Earlier releases used semver `v0.1.0`…`v0.9.0`, then one unpadded `26.33.0`
before this padding change.)

After publishing, the workflow regenerates **[CHANGELOG.md](CHANGELOG.md)** from
the release notes and opens an auto-merging PR into `develop` (since `main` can't
be pushed directly). The changelog reaches `main` on the next release. To
regenerate locally at any time: `./scripts/gen-changelog.sh`.

**Requires a `RELEASE_PAT` secret.** GitHub does not run workflows for events
made with the built-in `GITHUB_TOKEN`, so a bot-opened PR would never run the
required `test`/`branch-policy` checks and could never merge. A fine-grained PAT
(Contents + Pull requests: read/write on this repo) stored as the `RELEASE_PAT`
secret lets the changelog PR run its checks and auto-merge. Without the secret,
releases still succeed — the changelog step is skipped with a warning, and you
run `./scripts/gen-changelog.sh` in your next release PR instead.

## Reviews and housekeeping

- **[CODEOWNERS](.github/CODEOWNERS)** auto-requests the owner as reviewer on
  every PR. Review **approval is not required** to merge — on a single-maintainer
  repo that would deadlock, since you can't approve your own PR. Merges are gated
  on the `test` and `branch-policy` checks instead. To require approvals later,
  add a second collaborator, then set `required_approving_review_count` to 1.
- **Merged branches are auto-deleted** (repo setting `delete_branch_on_merge`),
  so work branches don't pile up.

## Adjusting the rules

The protection is applied with the GitHub API (see `scripts/setup-branch-protection.sh`).
To relax "admins included", set `enforce_admins=false` for a branch. To change
which checks are required, edit the `contexts` list.
