<!--
  Branching model: work → develop → main  (see BRANCHING.md)
  - PRs into `main` are only allowed from `develop`.
  - PRs into `develop` are only allowed from work branches.
  The `test` and `branch-policy` checks must pass before merging.
-->

## What & why

<!-- Describe the change and the motivation. Link any related issue (e.g. Closes #12). -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Docs / tooling
- [ ] Release (`develop` → `main`)

## Checklist

- [ ] `./gradlew test` passes locally
- [ ] For math/ops changes: `./gradlew run --args="gradcheck"` still reports **PASS**
- [ ] Docs updated (README / ARCHITECTURE / CONTRIBUTING) if behavior or structure changed
- [ ] New differentiable ops are gradient-checked (see CONTRIBUTING.md)

## Release notes (release PRs only)

<!--
  On a `develop` → `main` merge, a GitHub Release is tagged automatically as
  YY.WW.NN (ISO week-date, zero-padded patch, e.g. 26.34.01). No markers needed.
-->
