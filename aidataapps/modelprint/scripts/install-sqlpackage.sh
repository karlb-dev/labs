#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
tool_dir="$lab_dir/tools/sqlpackage"
if [[ -x "$tool_dir/sqlpackage" ]]; then
  "$tool_dir/sqlpackage" /Version
  exit 0
fi
command -v unzip >/dev/null || { apt-get update && apt-get install -y unzip libunwind8; }
archive="$(mktemp --suffix=.zip)"
trap 'rm -f "$archive"' EXIT
curl -fsSL https://aka.ms/sqlpackage-linux -o "$archive"
mkdir -p "$tool_dir"
unzip -q "$archive" -d "$tool_dir"
chmod a+x "$tool_dir/sqlpackage"
"$tool_dir/sqlpackage" /Version
