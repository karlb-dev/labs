#!/usr/bin/env bash

# Source this file from lab scripts after changing to the lab root. It loads the
# generated environment and selects the nested-Colab Docker endpoint/Compose
# override when that runtime profile is enabled.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]]; then
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/1000/docker.sock}"
  export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.colab.yaml}"
fi
