#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
tool_dir="$lab_dir/tools/sqlpackage"
if [[ -x "$tool_dir/sqlpackage" ]]; then
  "$tool_dir/sqlpackage" /Version
  exit 0
fi
case "$(uname -s)" in
  Linux)
    archive_url="https://aka.ms/sqlpackage-linux"
    if ! command -v unzip >/dev/null || ! ldconfig -p 2>/dev/null | grep -q 'libunwind\.so'; then
      command -v apt-get >/dev/null || { echo "Install unzip and libunwind before continuing." >&2; exit 1; }
      apt-get update
      apt-get install -y unzip libunwind8
    fi
    ;;
  Darwin)
    archive_url="https://aka.ms/sqlpackage-macos"
    command -v unzip >/dev/null || { echo "Install unzip before continuing." >&2; exit 1; }
    ;;
  *) echo "Unsupported SqlPackage host: $(uname -s)" >&2; exit 1 ;;
esac
archive="$(mktemp "${TMPDIR:-/tmp}/modelprint-sqlpackage.XXXXXX")"
trap 'rm -f "$archive"' EXIT
curl -fsSL "$archive_url" -o "$archive"
mkdir -p "$tool_dir"
unzip -q "$archive" -d "$tool_dir"
chmod a+x "$tool_dir/sqlpackage"
"$tool_dir/sqlpackage" /Version
