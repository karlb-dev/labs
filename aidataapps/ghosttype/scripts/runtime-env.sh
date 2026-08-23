#!/usr/bin/env bash

# Source from the Lab 4 root: loads .env and selects the compose overlay.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]]; then
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/1000/docker.sock}"
  export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.colab.yaml}"
elif [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "mac-docker-desktop" ]]; then
  export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.mac.yaml}"
fi
