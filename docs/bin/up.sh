#!/usr/bin/env bash


set -e -u -o pipefail
NEXT_VERSION=0.0.22
DEV_NAME='dev1'

if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

cd "$(dirname "$0")/../"
CURRENT_VERSION=$(sed -n 's/current_version:\s*//p' _config.yml)
python3 bin/freezeAtVersion.py --version "${CURRENT_VERSION}"
sed -i 's/\(current_version:\).*/\1 '"${NEXT_VERSION}"'/' _config.yml
sed -i '1i- title: '"${CURRENT_VERSION}" _data/previous_versions.yml
UNDER=$(echo $CURRENT_VERSION | tr . _)
sed -i '2i\ \ url: /docs/'"${UNDER}" _data/previous_versions.yml

sed -i 's/'"\(version = u\).*/\1'${NEXT_VERSION}'/" ../keanu-python/docs/conf.py
sed -i 's/'"\(release = u\).*/\1'${NEXT_VERSION}'/" ../keanu-python/docs/conf.py
sed -i "s/\(__version__ = '\).*/\1${NEXT_VERSION}.${DEV_NAME}'/" ../keanu-python/keanu/__version__.py
sed -i "1i # Version '${NEXT_VERSION}'\n" ../release_notes.md
