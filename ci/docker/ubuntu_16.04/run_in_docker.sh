#!/usr/bin/env bash

set -e -u -o pipefail

cd "$(dirname "$0")/../../.."

ci/docker/utils/run_in_docker.sh ubuntu_16.04 "$@"