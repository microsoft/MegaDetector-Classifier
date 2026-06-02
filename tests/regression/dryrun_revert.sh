#!/usr/bin/env bash
# Revert dry-run for a single commit. Creates a repo-local sibling worktree,
# reverts the commit there, runs the regression suite, then cleans up.
#
# Usage: bash tests/regression/dryrun_revert.sh <SHA>
#
# Exits 0 if the regression suite passes after revert, non-zero otherwise.
# The worktree is removed on exit via trap regardless of exit status.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <SHA>" >&2
  exit 2
fi

SHA="$1"
REPO_ROOT="$(git rev-parse --show-toplevel)"
WT="$REPO_ROOT/.revert-dryrun-$$"

cleanup() {
  git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT" 2>/dev/null || true
}
trap cleanup EXIT

git worktree add "$WT" HEAD
pushd "$WT" > /dev/null
git revert --no-edit "$SHA"
uv run pytest tests/regression -x
popd > /dev/null
