#!/usr/bin/env bash
#
## This a reduced variant of the interface of
## https://brevi.link/platform-source-file/ci/lib/log.sh.

log_action() {
  echo "$@"
}

log_success() {
  echo "$@"
}

# Prints the given message and exits with status 1 (failure).
log_fatal() {
  echo "$@"
  exit 1
}
