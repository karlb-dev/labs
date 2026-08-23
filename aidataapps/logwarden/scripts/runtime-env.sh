#!/usr/bin/env bash

# Source this file from the Lab 3 root. It loads the ignored environment and
# selects the nested-Colab Docker endpoint/Compose overlay when requested.
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
  # Docker Desktop owns the socket through the default context; do not set
  # DOCKER_HOST. The overlay pins the SQL service to linux/amd64 (Rosetta).
  export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.mac.yaml}"
fi
