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

## Adjusting the rules

The protection is applied with the GitHub API (see `scripts/setup-branch-protection.sh`).
To relax "admins included", set `enforce_admins=false` for a branch. To change
which checks are required, edit the `contexts` list.
