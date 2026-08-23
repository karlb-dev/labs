#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

: "${SQLSERVER_BASE_IMAGE:?SQLSERVER_BASE_IMAGE is required}"
: "${SQLSERVER_FTS_IMAGE:?SQLSERVER_FTS_IMAGE is required}"

docker pull "$SQLSERVER_BASE_IMAGE" >/dev/null
engine_version="$(docker image inspect "$SQLSERVER_BASE_IMAGE" --format '{{index .Config.Labels "com.microsoft.version"}}')"
if [[ ! "$engine_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "SQL base image has no valid com.microsoft.version label: $engine_version" >&2
  exit 3
fi
expected_package="${MSSQL_FTS_PACKAGE_VERSION:-${engine_version}-1}"
if [[ "$expected_package" != "${engine_version}-1" ]]; then
  echo "FTS package $expected_package does not match engine $engine_version" >&2
  exit 3
fi

# Nested rootless Docker cannot mount a fresh /proc during a Dockerfile RUN.
# Install in a live, host-namespace build container, commit it while running,
# and restore the exact base entrypoint/CMD/user. The resulting image receives
# the same runtime probe as a normal Dockerfile build.
build_container="logwarden-fts-build-$$"
cleanup() {
  docker rm --force "$build_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --detach \
  --name "$build_container" \
  --user root \
  --network host \
  --pid host \
  --ipc host \
  --cgroupns host \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --mount type=bind,src=/proc,dst=/proc,readonly \
  --mount type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly \
  --env "EXPECTED_FTS_PACKAGE=$expected_package" \
  --entrypoint /bin/bash \
  "$SQLSERVER_BASE_IMAGE" \
  -lc '
    set -Eeuo pipefail
    wget -qO /etc/apt/sources.list.d/mssql-server-2025.list \
      https://packages.microsoft.com/config/ubuntu/24.04/mssql-server-2025.list
    apt-get update
    candidate="$(apt-cache policy mssql-server-fts | awk "/Candidate:/ {print \$2}")"
    if [[ "$candidate" != "$EXPECTED_FTS_PACKAGE" ]]; then
      echo "Expected FTS package $EXPECTED_FTS_PACKAGE, repository offered $candidate" >&2
      exit 4
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      "mssql-server-fts=$EXPECTED_FTS_PACKAGE"
    dpkg-query -W -f="\${Package}=\${Version}\n" mssql-server-fts
    rm -rf /var/lib/apt/lists/*
    touch /tmp/logwarden-fts-ready
    exec sleep infinity
  ' >/dev/null

ready=0
for _ in $(seq 1 120); do
  if docker exec "$build_container" test -f /tmp/logwarden-fts-ready 2>/dev/null; then
    ready=1
    break
  fi
  if [[ "$(docker inspect "$build_container" --format '{{.State.Running}}')" != "true" ]]; then
    docker logs "$build_container" >&2
    exit 4
  fi
  sleep 1
done
if ((ready == 0)); then
  docker logs "$build_container" >&2
  echo "Timed out installing SQL Server full-text package" >&2
  exit 4
fi

docker commit --pause=false \
  --change 'USER mssql' \
  --change 'ENTRYPOINT ["/opt/mssql/bin/launch_sqlservr.sh"]' \
  --change 'CMD ["/opt/mssql/bin/sqlservr"]' \
  --change "LABEL ai.labs.base-image=$SQLSERVER_BASE_IMAGE" \
  --change "LABEL ai.labs.fts-package=$expected_package" \
  "$build_container" "$SQLSERVER_FTS_IMAGE" >/dev/null

image_id="$(docker image inspect "$SQLSERVER_FTS_IMAGE" --format '{{.Id}}')"
installed="$(docker exec "$build_container" dpkg-query -W -f='${Version}' mssql-server-fts)"
if [[ "$installed" != "$expected_package" ]]; then
  echo "Committed FTS package drift: expected $expected_package, saw $installed" >&2
  exit 4
fi
printf 'Built %s (%s) from engine %s with mssql-server-fts=%s\n' \
  "$SQLSERVER_FTS_IMAGE" "$image_id" "$engine_version" "$installed"
