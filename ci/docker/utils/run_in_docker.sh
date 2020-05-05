#!/usr/bin/env bash

set -e -u -o pipefail

cd "$(dirname "$0")/../../.."

source ci/docker/utils/helpers.sh

DOCKERFILE_DIRECTORY="$1"
shift

# Ensure that the required Docker image is present.
buildDockerImage "${DOCKERFILE_DIRECTORY}" # Exports IMAGE_IDENTIFIER.

MOUNT_PATH="$(realpath $(pwd))"

CACHE_PATH="/tmp/build_output/${BUILDKITE_AGENT_ID:-}/${IMAGE_IDENTIFIER//:/-}"
mkdir --parents "${CACHE_PATH}"

if isBuildKite; then
  ADDITIONAL_FLAGS=(
    --volume "/usr/bin/buildkite-agent:/usr/bin/buildkite-agent"
    --env BUILDKITE="${BUILDKITE}"
    --env BUILDKITE_AGENT_ACCESS_TOKEN="${BUILDKITE_AGENT_ACCESS_TOKEN}"
    --env BUILDKITE_BRANCH="${BUILDKITE_BRANCH}"
    --env BUILDKITE_BUILD_ID="${BUILDKITE_BUILD_ID}"
    --env BUILDKITE_BUILD_NUMBER="${BUILDKITE_BUILD_NUMBER}"
    --env BUILDKITE_COMMIT="${BUILDKITE_COMMIT}"
    --env BUILDKITE_JOB_ID="${BUILDKITE_JOB_ID}"
    --env BUILDKITE_LABEL="${BUILDKITE_LABEL}"
    --env BUILDKITE_PIPELINE_SLUG="${BUILDKITE_PIPELINE_SLUG}"
    --env BUILDKITE_RETRY_COUNT="${BUILDKITE_RETRY_COUNT}"
    --env BUILDKITE_AGENT_ENDPOINT="${BUILDKITE_AGENT_ENDPOINT}"
  )
fi
# Check if stdout is connected to a TTY.
if [[ -t 1 ]]; then
  # Passes through keyboard input, i.e. allows Ctrl+C to stop the container
  TTY_FLAGS="--tty --interactive"
else
  TTY_FLAGS=""
fi

USER_ID=$(id -u)
GROUP_ID=$(id -g)
GROUPS_IDS=$(id -G)

docker run \
  ${TTY_FLAGS} \
  --rm `# Removes the image when container is shut down` \
  --privileged \
  --volume "${MOUNT_PATH}:${MOUNT_PATH}" \
  --volume "${CACHE_PATH}:/tmp/build_output" \
  --volume "/var/node-metrics/text-files:/nfr-metrics" \
  \
  `# Needed to pass through the ssh-agent` \
  --volume "${SSH_AUTH_SOCK}:/tmp/ssh.sock" \
  --env "SSH_AUTH_SOCK=/tmp/ssh.sock" \
  --env "CI_ENVIRONMENT=${CI_ENVIRONMENT:-production}" \
  --env "AUTH_TOKEN=$(gcloud auth application-default print-access-token)" \
  \
  --env "USER_ID=${USER_ID}" \
  --env "GROUP_ID=${GROUP_ID}" \
  --env "GROUPS_IDS=${GROUPS_IDS}" \
  \
  "${ADDITIONAL_FLAGS[@]+"${ADDITIONAL_FLAGS[@]}"}" \
  \
  --workdir "${MOUNT_PATH}" \
  --cap-add=SYS_PTRACE `# Needed to use clang sanitizers` \
  ${IMAGE_IDENTIFIER} \
  "$@"
