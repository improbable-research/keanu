#!/usr/bin/env bash
# https://brevi.link/shell-style
# https://explainshell.com

if [[ ! `git status` ]]; then
  echo 'Not in git repo, exiting'
  exit 1
fi

repo_top_level_dir=`git rev-parse --show-toplevel`
cd ${repo_top_level_dir}/java

# clone dependencies
dependency='keanu'
rm -rf ./var/${dependency}
git clone "git@github.com:improbable-research/${dependency}.git" "./var/${dependency}"

docker build -t gradle-image .
exec docker run \
    --rm \
    gradle-image \
    ./gradlew clean build