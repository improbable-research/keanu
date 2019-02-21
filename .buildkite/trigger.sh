#!/usr/bin/env bash
# https://brevi.link/shell-style
# https://explainshell.com
set -euo pipefail
[[ -n "${DEBUG-}" ]] && set -x
cd "$(dirname "$0")/../"

# The step-definitions file is uploaded dynamically to preserve ability for historical builds
# vs changes in CI pipeline configuration.
buildkite-agent pipeline upload "$1"
