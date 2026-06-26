#!/usr/bin/env bash
# Reject commit messages with a body — keep the git log to one line per commit.
set -euo pipefail

file="${1:?commit message file required}"
non_empty="$(grep -cve '^#' -e '^[[:space:]]*$' "${file}" || true)"

if [ "${non_empty}" -gt 1 ]; then
  echo "Commit message must be a single line (conventional commit subject only)."
  echo "Add details in the pull request description instead."
  exit 1
fi
