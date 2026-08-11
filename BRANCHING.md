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
automatically tags a semver release and publishes a GitHub Release with
auto-generated notes. The bump is derived from the release PR's title:

| Marker in PR title | Result (from `vX.Y.Z`) |
|--------------------|------------------------|
| _(none)_           | patch → `vX.Y.(Z+1)`   |
| `[minor]`          | `vX.(Y+1).0`           |
| `[major]`          | `v(X+1).0.0`           |

The first release is `v0.1.0`. Tags are created on `main`; no manual tagging needed.

After publishing, the workflow regenerates **[CHANGELOG.md](CHANGELOG.md)** from
the release notes and opens an auto-merging PR into `develop` (since `main` can't
be pushed directly). The changelog reaches `main` on the next release. To
regenerate locally: `./scripts/gen-changelog.sh`.

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
