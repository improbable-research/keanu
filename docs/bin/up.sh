#!/usr/bin/env bash
set -e -u -o pipefail
CURRENT_VERSION=0.0.21
NEXT_VERSION=0.0.22
 if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

cd "$(dirname "$0")/../"
python3 bin/freezeAtVersion.py --version ${CURRENT_VERSION}
sed -ri 's/^(\s*)(current_version\s*:\s*'"${CURRENT_VERSION}"'\s*$)/\1current_version: '"${NEXT_VERSION}"'/' _config.yml
sed -i '1i- title: '"${CURRENT_VERSION}" _data/previous_versions.yml
UNDER=`echo $CURRENT_VERSION | tr . _`
sed -i '2i\ \ url: /docs/'"${UNDER}" _data/previous_versions.yml
