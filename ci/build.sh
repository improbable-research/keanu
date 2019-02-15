#!/usr/bin/env bash
set -e -u -o pipefail

if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

cd "$(dirname "$0")/../"

./gradlew build -x generateDocumentation -x deleteUnneededJarsFromPythonClasspath
