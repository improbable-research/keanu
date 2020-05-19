#!/usr/bin/env bash

# usage: getImageName <Dockerfile directory under ci/docker>
# i.e. getImageName ubuntu_16.04
function getImageName() {
    DOCKERFILE_DIRECTORY="$1"
    echo "eu.gcr.io/windy-oxide-102215/keanu-${DOCKERFILE_DIRECTORY}"
}

function isBuildKite() {
  # BUILDKITE="true" within Buildkite.
  [[ -n ${BUILDKITE+x} ]];
}

# usage: getContextSha
# Takes no arguments
function getContextSha() {
    echo "$(find ci/docker/ -type f | sort | xargs sha256sum | sha256sum | awk '{print $1}')"
}

# usage: buildDockerImage <Dockerfile directory under ci/docker>
# i.e. buildDockerImage ubuntu_16.04
function buildDockerImage() {
  DOCKERFILE_DIRECTORY="$1"

  local IMAGE_NAME="$(getImageName ${DOCKERFILE_DIRECTORY})"
  # We're using the sha of the context as the version of the docker container.
  local CONTEXT_SHA="$(getContextSha)"
  # i.e. keanu-ubuntu_16.04:87a364fbcbe476252a667d2141d77dba123b15dea51385ca969aeebe0a843358
  export IMAGE_IDENTIFIER="${IMAGE_NAME}:${CONTEXT_SHA}"

  IMAGE_PRESENT=false

  # Check for image existence, first locally, then remotely.
  if [[ "$(docker images "${IMAGE_IDENTIFIER}" | wc -l)" -eq 2 ]]; then
    echo "Image present locally"
    IMAGE_PRESENT=true

  elif docker pull "${IMAGE_IDENTIFIER}"; then
    echo "Image present remotely"
    IMAGE_PRESENT=true
  fi

  # Build the container if it isn't present.
  if [[ "${IMAGE_PRESENT}" = false ]]; then
    docker image build \
      --build-arg=IMAGE_IDENTIFIER="${IMAGE_IDENTIFIER}" \
      --tag="${IMAGE_IDENTIFIER}" ci/docker \
      --file="ci/docker/${DOCKERFILE_DIRECTORY}/Dockerfile"

    # Push the image if we're running in a BuildKite agent with pusher permissions.
    if isBuildKite && [[ "${BUILDKITE_AGENT_META_DATA_PERMISSION_SET}" = "pusher" ]]; then
      echo "Pushing image"
      docker push ${IMAGE_IDENTIFIER}
    fi
  fi

}
