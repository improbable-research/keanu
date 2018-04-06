#!/usr/bin/env bash
#
## Builds in preparation for testing and deployment.

source "$(dirname "${BASH_SOURCE[0]}")/lib/setup.sh" || exit 1

gradle build -i
