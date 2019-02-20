#!/usr/bin/env bash

set -e -u -o pipefail

if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

PYPI_TEST_SECRET="ci/improbable/keanu-eng-test-pypi"
PYPI_REAL_SECRET="ci/improbable/keanu-eng-test-pypi"

display_usage() {
	echo
	echo "Usage: $0 {PYPI_REPO} {RELEASE_TYPE}"
	echo "PYPI_REPO options: test real"
	echo "RELEASE_TYPE options: ci manual"
	echo
}

get_secrets_for_manual_release() {
	PYPI_USERNAME=$(imp-vault read --product_group="dev-workflow" --environment="production" --role="buildkite" --in_use_by="buildkite-agents" --type_name="generic-credentials" --name=${secret_name} --field="username")
    PYPI_PASSWORD=$(imp-vault read --product_group="dev-workflow" --environment="production" --role="buildkite" --in_use_by="buildkite-agents" --type_name="generic-credentials" --name=${secret_name} --field="password")
}

get_secrets_for_ci_release() {
	PYPI_USERNAME=$(imp-ci secrets read --environment="production" --buildkite-org="improbable" --secret-type="generic-credentials" --secret-name=${secret_name} --field="username")
	PYPI_PASSWORD=$(imp-ci secrets read --environment="production" --buildkite-org="improbable" --secret-type="generic-credentials" --secret-name=${secret_name} --field="password")
}

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    display_usage
    exit 1
fi

target_pypi_arg=${1}

case $target_pypi_arg in
	test)
      secret_name=$PYPI_TEST_SECRET
      ;;
    real)
      secret_name=$PYPI_REAL_SECRET
      ;;
    *)
      echo "Unknown target repo"
      display_usage
      exit 1
      ;;
esac

release_type=${2}

case $release_type in
	manual)
      get_secrets_for_manual_release
      ;;
    ci)
      get_secrets_for_ci_release
      ;;
    *)
      echo "Unknown release type"
      display_usage
      exit 1
      ;;
esac


cd "$(dirname "$0")/../"

python3 -m twine upload --username "${PYPI_USERNAME}" --password "${PYPI_PASSWORD}" --repository-url https://test.pypi.org/legacy/ keanu-python/dist/*
