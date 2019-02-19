#!/usr/bin/env bash
## Script upgrades the version numbers in the places necessary across the repo.
## It will add the current and next version to appropiate locations.
## Used for after a release to start work on next version.

set -e -u -o pipefail
NEXT_VERSION=0.0.22
DEV_NAME='dev1'

if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

cd "$(dirname "$0")/../"

# Get the current version from _config.yml
CURRENT_VERSION=$(sed -n 's/current_version:\s*//p' _config.yml)

# Freeze Python and Shiny docs
python3 bin/freezeAtVersion.py --version "${CURRENT_VERSION}"

# Update version in _config.yml
sed -i 's/\(current_version:\).*/\1 '"${NEXT_VERSION}"'/' _config.yml

# Update previous_versions.yml
UNDERSCORED_CURRENT_VERSION=$(echo "${CURRENT_VERSION}" | tr . _)
sed -i '1i- title: '"${CURRENT_VERSION}" _data/previous_versions.yml
sed -i '2i\ \ url: /docs/'"${UNDERSCORED_CURRENT_VERSION}" _data/previous_versions.yml

# Update conf.py version
sed -i 's/'"\\(version = u\\).*/\\1'${NEXT_VERSION}'/" ../keanu-python/docs/conf.py
sed -i 's/'"\\(release = u\\).*/\\1'${NEXT_VERSION}'/" ../keanu-python/docs/conf.py

# Update Python __version__
sed -i "s/\\(__version__ = '\\).*/\\1${NEXT_VERSION}.${DEV_NAME}'/" ../keanu-python/keanu/__version__.py

# Add new version number to release notes
sed -i "1i # Version ${NEXT_VERSION}\\n" ../release_notes.md
