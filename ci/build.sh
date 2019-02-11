#!/usr/bin/env bash
# https://brevi.link/shell-style
# https://explainshell.com

if [[ ! `git status` ]]; then
  echo 'Not in git repo, exiting'
  exit 1
fi

repo_top_level_dir=`git rev-parse --show-toplevel`
cd ${repo_top_level_dir}/java

docker build -t build-image -f docker/builder/Dockerfile .
exec docker run \
    --rm \
    build-image \
    ./gradlew clean build