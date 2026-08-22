#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this bootstrap as root inside the Colab VM." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
docker_user="${RAG_DOCKER_USER:-codexdocker}"
docker_uid="${RAG_DOCKER_UID:-1000}"
runtime_dir="/run/user/$docker_uid"
docker_socket="unix://$runtime_dir/docker.sock"
daemon_log="${RAG_DOCKER_LOG:-/content/aidataapps-rag-dockerd.log}"

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This bootstrap currently supports Ubuntu Colab runtimes, not ${ID:-unknown}." >&2
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl fuse-overlayfs gnupg openssl uidmap slirp4netns

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
printf '%s\n' \
  'Types: deb' \
  'URIs: https://download.docker.com/linux/ubuntu' \
  "Suites: ${VERSION_CODENAME}" \
  'Components: stable' \
  'Signed-By: /etc/apt/keyrings/docker.asc' \
  >/etc/apt/sources.list.d/docker.sources

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  >/etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y \
  docker-ce docker-ce-cli docker-buildx-plugin docker-compose-plugin \
  docker-ce-rootless-extras nvidia-container-toolkit

if ! id "$docker_user" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash --uid "$docker_uid" "$docker_user"
fi

if ! grep -q "^${docker_user}:" /etc/subuid; then
  usermod --add-subuids 100000-165535 "$docker_user"
fi
if ! grep -q "^${docker_user}:" /etc/subgid; then
  usermod --add-subgids 100000-165535 "$docker_user"
fi

install -d -m 0755 /dev/net
if [[ ! -e /dev/net/tun ]]; then
  mknod /dev/net/tun c 10 200
fi
chmod 0666 /dev/net/tun

# Colab itself is a container. Its cgroup mount starts read-only even though
# the notebook process is root; rootless runc needs this delegated mount.
if ! findmnt -no OPTIONS /sys/fs/cgroup | grep -qw rw; then
  mount -o remount,rw /sys/fs/cgroup
fi

install -d -m 0700 -o "$docker_user" -g "$docker_user" "$runtime_dir"
install -d -m 0755 -o "$docker_user" -g "$docker_user" \
  "/home/$docker_user/.config/docker"
nvidia-ctk runtime configure \
  --runtime=docker \
  --config="/home/$docker_user/.config/docker/daemon.json"
chown -R "$docker_user:$docker_user" "/home/$docker_user/.config"

# Rootless runtimes cannot manage device cgroups. CDI supplies the devices and
# mounts explicitly instead.
nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
if [[ -d /usr/lib64-nvidia ]]; then
  echo /usr/lib64-nvidia >/etc/ld.so.conf.d/colab-nvidia.conf
  ldconfig
fi
install -d -m 0755 /etc/cdi
nvidia-ctk cdi generate --disable-hook update-ldcache \
  --output=/etc/cdi/nvidia.yaml

if ! DOCKER_HOST="$docker_socket" docker info >/dev/null 2>&1; then
  nohup runuser -u "$docker_user" -- env \
    HOME="/home/$docker_user" \
    XDG_RUNTIME_DIR="$runtime_dir" \
    PATH=/usr/bin:/sbin:/usr/sbin:/usr/local/bin \
    dockerd-rootless.sh --storage-driver=fuse-overlayfs \
    >"$daemon_log" 2>&1 &
fi

for _ in $(seq 1 60); do
  if DOCKER_HOST="$docker_socket" docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
DOCKER_HOST="$docker_socket" docker info >/dev/null

if [[ ! -f "$lab_dir/.env" ]]; then
  generated_password="RagLab!$(openssl rand -hex 18)"
  sed "s/replace-with-a-generated-strong-password/$generated_password/" \
    "$lab_dir/.env.example" >"$lab_dir/.env"
  chmod 0600 "$lab_dir/.env"
fi
profile_tmp="$(mktemp "$lab_dir/.env.tmp.XXXXXX")"
awk '
  BEGIN { replaced = 0 }
  /^CONTAINER_RUNTIME_PROFILE=/ {
    print "CONTAINER_RUNTIME_PROFILE=colab-rootless"
    replaced = 1
    next
  }
  { print }
  END {
    if (!replaced) print "CONTAINER_RUNTIME_PROFILE=colab-rootless"
  }
' "$lab_dir/.env" >"$profile_tmp"
chmod 0600 "$profile_tmp"
mv "$profile_tmp" "$lab_dir/.env"

echo "Rootless Docker is ready at $docker_socket"
echo "Daemon log: $daemon_log"
echo "Next: ./scripts/env-init.sh"
