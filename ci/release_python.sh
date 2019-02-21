#!/usr/bin/env bash

set -e -u -o pipefail

if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

cd "$(dirname "$0")/"

PYPI_TEST_SECRET="keanu-eng-test-pypi"
PYPI_REAL_SECRET="keanu-eng-test-pypi"

display_usage() {
	echo
	echo "Usage: $0 {PYPI_REPO} {RELEASE_TYPE}"
	echo "PYPI_REPO options: test real"
	echo "RELEASE_TYPE options: ci manual"
	echo
}

get_secrets() {
  release_type=$1
  secret_name=$2
  case ${release_type} in
  manual)
      pypi_username=$(imp-vault read --product_group="dev-workflow" --environment="production" --role="buildkite" --in_use_by="buildkite-agents" --type_name="generic-credentials" --name="ci/improbable/${secret_name}" --field="username")
      pypi_password=$(imp-vault read --product_group="dev-workflow" --environment="production" --role="buildkite" --in_use_by="buildkite-agents" --type_name="generic-credentials" --name="ci/improbable/${secret_name}" --field="password")
      ;;
    ci)
      pypi_username=$(imp-ci secrets read --environment="production" --buildkite-org="improbable" --secret-type="generic-credentials" --secret-name="${secret_name}" --field="username")
      pypi_password=$(imp-ci secrets read --environment="production" --buildkite-org="improbable" --secret-type="generic-credentials" --secret-name="${secret_name}" --field="password")
      
      # Delete dist directory if it already exists.
      rm -rf ../keanu-python/dist/
      mkdir ../keanu-python/dist/
      # Download artifacts from Buildkite
      buildkite-agent artifact download --step "Build Python distribution" "keanu-python/dist/*" "../keanu-python/dist/"
      ;;
    *)
      echo "Unknown release type"
      display_usage
      exit 1
      ;;
esac
}

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    display_usage
    exit 1
fi

target_pypi_arg=${1}
release_type=${2}

pipenv install

case $target_pypi_arg in
	test)
      get_secrets "${release_type}" ${PYPI_TEST_SECRET}
      pipenv run python3 -m twine upload --username "${pypi_username}" --password "${pypi_password}" --repository-url https://test.pypi.org/legacy/ ../keanu-python/dist/*
      ;;
    real)
      get_secrets "${release_type}" ${PYPI_REAL_SECRET}
      pipenv run python3 -m twine upload --username "${pypi_username}" --password "${pypi_password}" ../keanu-python/dist/*
      ;;
    *)
      echo "Unknown target repo"
      display_usage
      exit 1
      ;;
esac
