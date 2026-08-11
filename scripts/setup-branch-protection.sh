#!/usr/bin/env bash
# Apply branch protection to main and develop for the Kortex branching model.
#
#   - require a pull request before merging (blocks direct pushes)
#   - require the "test" and "branch-policy" status checks to pass
#   - include administrators (direct pushes prohibited for everyone)
#
# Usage:  ./scripts/setup-branch-protection.sh [owner/repo]
# Requires: gh (authenticated with admin rights on the repo).
set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
echo "Applying branch protection to $REPO"

protect() {
  local branch="$1"
  echo "  -> $branch"
  gh api -X PUT "repos/$REPO/branches/$branch/protection" \
    --input - <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["test", "branch-policy"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
}

protect main
protect develop
echo "Done. Direct pushes are blocked; merges require passing checks."
