#!/usr/bin/env bash

container_mount_source() {
  local container="$1" destination="$2"
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]]; then
    local docker_root="${ROOTLESS_DOCKER_DATA_ROOT:-/home/codexdocker/.local/share/docker}" config
    for config in "$docker_root"/containers/*/config.v2.json; do
      [[ -f "$config" ]] || continue
      if jq -e --arg name "/$container" '.Name==$name' "$config" >/dev/null; then
        jq -er --arg destination "$destination" '.MountPoints[$destination].Source' "$config"
        return
      fi
    done
    echo "Cannot resolve rootless mount $destination for $container" >&2
    return 1
  fi
  docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.Source}}{{end}}{{end}}" "$container"
}

copy_from_container_file() {
  local container="$1" container_path="$2" target="$3"
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" && "$container_path" == /var/opt/mssql/* ]]; then
    local mount relative source
    mount="$(container_mount_source "$container" /var/opt/mssql)";relative="${container_path#/var/opt/mssql/}"
    [[ "$relative" != *".."* ]] || { echo "Unsafe container path" >&2; return 2; }
    source="$mount/$relative";[[ -f "$source" ]] || { echo "Missing container-volume file $source" >&2; return 2; }
    cp --reflink=auto "$source" "$target"
  else docker cp "$container:$container_path" "$target"; fi
}

copy_to_container_file() {
  local source="$1" container="$2" container_path="$3"
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" && "$container_path" == /var/opt/mssql/* ]]; then
    local mount relative target
    mount="$(container_mount_source "$container" /var/opt/mssql)";relative="${container_path#/var/opt/mssql/}"
    [[ "$relative" != *".."* ]] || { echo "Unsafe container path" >&2; return 2; }
    target="$mount/$relative";cp --reflink=auto "$source" "$target";chmod a+r "$target"
  else
    # docker cp leaves the file root-owned; the mssql user must be able to read it.
    docker cp "$source" "$container:$container_path" && docker exec -u root "$container" chmod a+r "$container_path"
  fi
}

remove_container_file() {
  local container="$1" container_path="$2"
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" && "$container_path" == /var/opt/mssql/data/* ]]; then
    local mount relative target
    mount="$(container_mount_source "$container" /var/opt/mssql)";relative="${container_path#/var/opt/mssql/}"
    [[ "$relative" != *".."* && "$relative" == data/* ]] || { echo "Unsafe container path" >&2; return 2; }
    target="$mount/$relative";[[ -f "$target" ]] && rm -f -- "$target"
  else docker exec "$container" rm -f -- "$container_path"; fi
}
