#!/usr/bin/env bash
# Top-level wrapper around tests/regression/dryrun_revert.sh.
# Resolves a phase SHA and runs the revert dry-run sandbox.
#
# Usage:
#   scripts/rollback_dry_run.sh <SHA>
#       Revert the specified commit in a sibling worktree and run
#       the regression suite. Worktree is cleaned up on exit.
#
# Future enhancement: also accept a phase number (0..9) and resolve to the
# corresponding `impl-dinov3-p<N>-merged` tag (no such tags exist in P0).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <SHA>" >&2
  exit 2
fi

SHA="$1"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Validate SHA exists
if ! git -C "$REPO_ROOT" cat-file -e "${SHA}^{commit}" 2>/dev/null; then
  echo "Error: ${SHA} is not a valid commit in this repo." >&2
  exit 3
fi

exec bash "$REPO_ROOT/tests/regression/dryrun_revert.sh" "$SHA"
